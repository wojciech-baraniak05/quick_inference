"""Wczesny stop oparty na marginesie lasu: Barbareschi, Emmanuele, "A Margin Based
Early-Stopping Approach for Random Forest Classifiers" (DSN-W 2025).
Opis: implementacja.md, rozdz. 7."""
from typing import Literal, Self, get_args

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier

from .base import FixedVoting, FloatArray, IntArray
from .oob import predykcje_oob, probka_uczaca

Ranking = Literal["mean_d_m", "accuracy", "none"]
RANKINGS: tuple[Ranking, ...] = get_args(Ranking)


def predykcje_kalibracyjne(rf: RandomForestClassifier, X_train: ArrayLike, y_train: ArrayLike,
                           n_calib: int | None = None, seed: int = 0) -> tuple[IntArray, IntArray]:
    """Zbiór kalibracyjny z predykcji out-of-bag: (pred, y), gdzie pred[i, n] to indeks
    klasy drzewa i dla próbki n (-1: drzewo widziało próbkę), a y[n] to klasa prawdziwa."""
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    if len(y_train) != len(X_train):
        raise ValueError(f"X_train ma {len(X_train)} wierszy, a y_train {len(y_train)}")
    wiersze = probka_uczaca(rf, X_train, n_calib, seed)
    y = np.searchsorted(rf.classes_, y_train[wiersze])
    pred = np.full((len(rf.estimators_), len(wiersze)), -1, dtype=np.int16)
    for i, (oob, p) in enumerate(predykcje_oob(rf, X_train, wiersze)):
        pred[i, oob] = p
    ok = (pred >= 0).sum(axis=0) >= 2       # margines bez jednego drzewa musi być określony
    return pred[:, ok], y[ok]


def glosy_z_predykcji(pred: IntArray, l: int) -> IntArray:
    """votes[n, c] = ile drzew głosuje na klasę c dla próbki n (-1 w pred pomijane)."""
    return np.stack([(pred == c).sum(axis=0) for c in range(l)], axis=1)


def margines(votes: IntArray, y: IntArray) -> FloatArray:
    """(głosy klasy prawdziwej - głosy najlepszej innej klasy) / liczba głosów."""
    wiersze = np.arange(len(votes))
    inne = votes.copy()
    inne[wiersze, y] = -1                           # klasa prawdziwa poza konkursem
    return (votes[wiersze, y] - inne.max(axis=1)) / votes.sum(axis=1)


def statystyki_marginesu(votes: IntArray, y: IntArray) -> tuple[float, float]:
    """(mu, sigma) marginesu próbek poprawnie sklasyfikowanych przez głosowanie."""
    poprawne = votes.argmax(axis=1) == y
    if not poprawne.any():
        raise ValueError("żadna próbka kalibracyjna nie jest poprawnie sklasyfikowana")
    mg = margines(votes[poprawne], y[poprawne])
    return float(mg.mean()), float(mg.std())


def liczba_drzew(T: int, mu: float, sigma: float, k: float) -> int:
    """N* = T * (mu + k * sigma), zaokrąglone i ograniczone do [1, T]."""
    return int(np.clip(np.rint(T * (mu + k * sigma)), 1, T))


def spadek_marginesu(pred: IntArray, y: IntArray, l: int) -> FloatArray:
    """Kryterium MeanD-M: średni spadek marginesu po usunięciu drzewa (-inf: brak głosów)."""
    votes = glosy_z_predykcji(pred, l)
    mg = margines(votes, y)
    # skutek usunięcia drzewa zależy tylko od klasy, na którą głosowało,
    # więc spadek liczymy raz dla pary (próbka, klasa), a nie dla każdego drzewa
    spadek = np.empty((len(y), l))
    for c in range(l):
        bez = votes.copy()
        bez[:, c] -= 1          # tam, gdzie nikt nie głosował na c, wynik nie jest czytany
        spadek[:, c] = mg - margines(bez, y)
    wynik = np.full(len(pred), -np.inf)
    for i, p in enumerate(pred):
        glosuje = np.flatnonzero(p >= 0)
        if len(glosuje):
            wynik[i] = spadek[glosuje, p[glosuje]].mean()
    return wynik


def trafnosc_drzew(pred: IntArray, y: IntArray) -> FloatArray:
    """Trafność każdego drzewa na próbkach, na które głosuje (-inf, gdy na żadną)."""
    glosuje = pred >= 0
    ile = glosuje.sum(axis=1)
    trafne = ((pred == y) & glosuje).sum(axis=1)
    return np.where(ile > 0, trafne / np.maximum(ile, 1), -np.inf)


def ranking_drzew(pred: IntArray, y: IntArray, l: int, ranking: Ranking = "mean_d_m") -> IntArray:
    """Indeksy drzew od najważniejszego; remisy rozstrzyga kolejność lasu."""
    if ranking == "none":
        return np.arange(len(pred))
    if ranking == "mean_d_m":
        ocena = spadek_marginesu(pred, y, l)
    elif ranking == "accuracy":
        ocena = trafnosc_drzew(pred, y)
    else:
        raise ValueError(f"nieznany ranking {ranking!r}, dostępne: {RANKINGS}")
    return np.argsort(-ocena, kind="stable")


class MarginEarlyStop(FixedVoting):
    """Głosuje N* = T * (mu + k * sigma) pierwszych drzew rankingu, tyle samo dla każdej próbki."""

    order: IntArray             # ranking drzew
    mu: float
    sigma: float
    n_star: int

    def __init__(self, k: float = 0.0, ranking: Ranking = "mean_d_m",
                 n_calib: int | None = 20000, name: str | None = None) -> None:
        if ranking not in RANKINGS:
            raise ValueError(f"MarginEarlyStop: nieznany ranking {ranking!r}, dostępne: {RANKINGS}")
        self.k, self.ranking, self.n_calib = k, ranking, n_calib
        super().__init__(name or f"MARGIN_k{k:g}")

    def fit(self, rf: RandomForestClassifier, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None) -> Self:
        super().fit(rf)
        if X_train is None or y_train is None:
            raise ValueError(f"{self.name}: fit(rf, X_train, y_train) wymaga danych uczących "
                             f"lasu z etykietami (kalibracja marginesu)")
        pred, y = predykcje_kalibracyjne(rf, X_train, y_train, self.n_calib)
        self.order = ranking_drzew(pred, y, self.l, self.ranking)
        self.mu, self.sigma = statystyki_marginesu(glosy_z_predykcji(pred, self.l), y)
        return self.set_k(self.k)

    def set_k(self, k: float) -> Self:
        """Zmienia k i liczbę głosujących drzew bez ponownej kalibracji (nazwa zostaje)."""
        self.k = k
        self.n_star = liczba_drzew(self.T, self.mu, self.sigma, k)
        self.voters = [self.trees[i] for i in self.order[:self.n_star]]
        return self
