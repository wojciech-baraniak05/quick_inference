"""Wczesny stop oparty na marginesie lasu: Barbareschi, Emmanuele, "A Margin Based
Early-Stopping Approach for Random Forest Classifiers" (DSN-W 2025).
Opis: implementacja.md, rozdz. 7."""
from typing import Literal, Self, get_args

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier

from .base import CALIB_SUFFIX, Calib, FixedVoting, FloatArray, IntArray, check_calib
from .oob import oob_predictions, sample_rows
from .validation import val_predictions

Ranking = Literal["mean_d_m", "accuracy", "none"]
RANKINGS: tuple[Ranking, ...] = get_args(Ranking)


def calibration_predictions(rf: RandomForestClassifier, X_train: ArrayLike, y_train: ArrayLike,
                           n_calib: int | None = None, seed: int = 0) -> tuple[IntArray, IntArray]:
    """Zbiór kalibracyjny z predykcji out-of-bag: (pred, y), gdzie pred[i, n] to indeks
    klasy drzewa i dla próbki n (-1: drzewo widziało próbkę), a y[n] to klasa prawdziwa."""
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    if len(y_train) != len(X_train):
        raise ValueError(f"X_train has {len(X_train)} rows, y_train {len(y_train)}")
    rows = sample_rows(rf, X_train, n_calib, seed)
    y = np.searchsorted(rf.classes_, y_train[rows])
    pred = np.full((len(rf.estimators_), len(rows)), -1, dtype=np.int16)
    for i, (oob, p) in enumerate(oob_predictions(rf, X_train, rows)):
        pred[i, oob] = p
    ok = (pred >= 0).sum(axis=0) >= 2       # margines bez jednego drzewa musi być określony
    return pred[:, ok], y[ok]


def votes_from_predictions(pred: IntArray, l: int) -> IntArray:
    """votes[n, c] = ile drzew głosuje na klasę c dla próbki n (-1 w pred pomijane)."""
    return np.stack([(pred == c).sum(axis=0) for c in range(l)], axis=1)


def vote_margin(votes: IntArray, y: IntArray) -> FloatArray:
    """(głosy klasy prawdziwej - głosy najlepszej innej klasy) / liczba głosów."""
    rows = np.arange(len(votes))
    others = votes.copy()
    others[rows, y] = -1                            # klasa prawdziwa poza konkursem
    return (votes[rows, y] - others.max(axis=1)) / votes.sum(axis=1)


def margin_stats(votes: IntArray, y: IntArray) -> tuple[float, float]:
    """(mu, sigma) marginesu próbek poprawnie sklasyfikowanych przez głosowanie."""
    correct = votes.argmax(axis=1) == y
    if not correct.any():
        raise ValueError("no calibration sample is classified correctly")
    mg = vote_margin(votes[correct], y[correct])
    return float(mg.mean()), float(mg.std())


def n_trees_from_margin(T: int, mu: float, sigma: float, k: float) -> int:
    """N* = T * (mu + k * sigma), zaokrąglone i ograniczone do [1, T]."""
    return int(np.clip(np.rint(T * (mu + k * sigma)), 1, T))


def margin_drop(pred: IntArray, y: IntArray, l: int) -> FloatArray:
    """Kryterium MeanD-M: średni spadek marginesu po usunięciu drzewa (-inf: brak głosów)."""
    votes = votes_from_predictions(pred, l)
    mg = vote_margin(votes, y)
    # skutek usunięcia drzewa zależy tylko od klasy, na którą głosowało,
    # więc spadek liczymy raz dla pary (próbka, klasa), a nie dla każdego drzewa
    drops = np.empty((len(y), l))
    for c in range(l):
        without = votes.copy()
        without[:, c] -= 1      # tam, gdzie nikt nie głosował na c, wynik nie jest czytany
        drops[:, c] = mg - vote_margin(without, y)
    result = np.full(len(pred), -np.inf)
    for i, p in enumerate(pred):
        voting = np.flatnonzero(p >= 0)
        if len(voting):
            result[i] = drops[voting, p[voting]].mean()
    return result


def tree_accuracy(pred: IntArray, y: IntArray) -> FloatArray:
    """Trafność każdego drzewa na próbkach, na które głosuje (-inf, gdy na żadną)."""
    voting = pred >= 0
    count = voting.sum(axis=1)
    hits = ((pred == y) & voting).sum(axis=1)
    return np.where(count > 0, hits / np.maximum(count, 1), -np.inf)


def rank_trees(pred: IntArray, y: IntArray, l: int, ranking: Ranking = "mean_d_m") -> IntArray:
    """Indeksy drzew od najważniejszego; remisy rozstrzyga kolejność lasu."""
    if ranking == "none":
        return np.arange(len(pred))
    if ranking == "mean_d_m":
        score = margin_drop(pred, y, l)
    elif ranking == "accuracy":
        score = tree_accuracy(pred, y)
    else:
        raise ValueError(f"unknown ranking {ranking!r}, expected one of {RANKINGS}")
    return np.argsort(-score, kind="stable")


class MarginEarlyStop(FixedVoting):
    """Głosuje N* = T * (mu + k * sigma) pierwszych drzew rankingu, tyle samo dla każdej próbki.
    mu, sigma i ranking z predykcji out-of-bag albo (calib="val") na zbiorze walidacyjnym."""

    order: IntArray             # ranking drzew
    mu: float
    sigma: float
    n_star: int

    def __init__(self, k: float = 0.0, ranking: Ranking = "mean_d_m",
                 n_calib: int | None = 20000, name: str | None = None,
                 calib: Calib = "oob") -> None:
        if ranking not in RANKINGS:
            raise ValueError(f"{type(self).__name__}: unknown ranking {ranking!r}, "
                             f"expected one of {RANKINGS}")
        check_calib(type(self).__name__, calib)
        self.k, self.ranking, self.n_calib, self.calib = k, ranking, n_calib, calib
        super().__init__(name or f"MARGIN_k{k:g}{CALIB_SUFFIX[calib]}")

    def fit(self, rf: RandomForestClassifier, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None, X_val: ArrayLike | None = None,
            y_val: ArrayLike | None = None) -> Self:
        super().fit(rf)
        if self.calib == "val":
            if X_val is None or y_val is None:
                raise ValueError(f"{self.name}: calib='val' needs a labelled validation set in "
                                 f"fit(rf, X_train, y_train, X_val, y_val) "
                                 f"(benchmark.val_size > 0)")
            pred, y = val_predictions(rf, X_val, y_val, self.n_calib)
        else:
            if X_train is None or y_train is None:
                raise ValueError(f"{self.name}: fit(rf, X_train, y_train) needs the forest's "
                                 f"labelled training data (margin calibration)")
            pred, y = calibration_predictions(rf, X_train, y_train, self.n_calib)
        self.order = rank_trees(pred, y, self.l, self.ranking)
        self.mu, self.sigma = margin_stats(votes_from_predictions(pred, self.l), y)
        return self.set_k(self.k)

    def set_k(self, k: float) -> Self:
        """Zmienia k i liczbę głosujących drzew bez ponownej kalibracji (nazwa zostaje)."""
        self.k = k
        self.n_star = n_trees_from_margin(self.T, self.mu, self.sigma, k)
        self.voters = [self.trees[i] for i in self.order[:self.n_star]]
        return self
