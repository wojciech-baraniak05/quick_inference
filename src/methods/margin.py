"""Wczesny stop oparty na marginesie lasu: Barbareschi, Emmanuele, "A Margin Based
Early-Stopping Approach for Random Forest Classifiers" (DSN-W 2025).

Margines próbki x (rozdz. II-B):

    M_x = (v_c - v_c*) / N

gdzie v_c to liczba głosów klasy prawdziwej, v_c* liczba głosów najczęściej wybieranej
innej klasy, a N liczba głosujących drzew. M_x > 0 oznacza poprawną klasyfikację.

Faza offline (rozdz. III-A), na zbiorze kalibracyjnym T_off z etykietami:
  1. ranking drzew od najbardziej do najmniej istotnego (w artykule: MeanD-M),
  2. średnia mu i odchylenie standardowe sigma marginesu próbek poprawnie
     sklasyfikowanych, a z nich liczba drzew potrzebnych do inferencji:

         N* = N * (mu + k * sigma)

     k steruje kompromisem: większe k = więcej drzew, mniejsze k = krótsza inferencja.

Faza po wdrożeniu (rozdz. III-B): każda próbka jest oceniana przez N* pierwszych drzew
rankingu, a wynik to głosowanie większościowe tych drzew. Liczba drzew jest więc stała
dla wszystkich próbek (inaczej niż w IB pruning, HYPER i GLEE).

W artykule T_off to połowa zbioru testowego. Tutaj T_off to próbki uczące oceniane
out-of-bag (każdą próbkę oceniają tylko drzewa, które jej nie widziały), więc cały zbiór
testowy zostaje do pomiaru, tak jak dla pozostałych metod.
"""
import numpy as np

from .base import FixedVoting
from .oob import predykcje_oob, probka_uczaca

RANKINGS = ("mean_d_m", "accuracy", "none")


def predykcje_kalibracyjne(rf, X_train, y_train, n_calib=None, seed=0):
    """Zbiór kalibracyjny T_off z predykcji out-of-bag na danych uczących lasu.

    Zwraca (pred, y):
        pred[i, n]  indeks klasy, na którą drzewo i głosuje dla próbki n,
                    albo -1, jeśli drzewo widziało tę próbkę przy uczeniu (nie głosuje),
        y[n]        indeks klasy prawdziwej.
    Zostają tylko próbki z co najmniej dwoma głosami (margines po usunięciu jednego
    drzewa musi być określony). n_calib ogranicza liczbę próbek; None = wszystkie.
    """
    if len(y_train) != len(X_train):
        raise ValueError(f"X_train ma {len(X_train)} wierszy, a y_train {len(y_train)}")
    wiersze = probka_uczaca(rf, X_train, n_calib, seed)
    y = np.searchsorted(rf.classes_, np.asarray(y_train)[wiersze])
    pred = np.full((len(rf.estimators_), len(wiersze)), -1, dtype=np.int16)
    for i, (oob, p) in enumerate(predykcje_oob(rf, X_train, wiersze)):
        pred[i, oob] = p
    ok = (pred >= 0).sum(axis=0) >= 2
    return pred[:, ok], y[ok]


def glosy_z_predykcji(pred, l):
    """votes[n, c] = ile drzew głosuje na klasę c dla próbki n (wartości -1 w pred pomijane)."""
    return np.stack([(pred == c).sum(axis=0) for c in range(l)], axis=1)


def margines(votes, y):
    """Margines każdej próbki: (głosy klasy prawdziwej - głosy najlepszej innej klasy)
    / liczba głosów. votes ma kształt (n, l), y to indeksy klas prawdziwych."""
    wiersze = np.arange(len(votes))
    inne = votes.copy()
    inne[wiersze, y] = -1                           # klasa prawdziwa poza konkursem
    return (votes[wiersze, y] - inne.max(axis=1)) / votes.sum(axis=1)


def statystyki_marginesu(votes, y):
    """(mu, sigma): średnia i odchylenie standardowe marginesu próbek poprawnie
    sklasyfikowanych przez głosowanie (argmax, czyli tak jak w full_voting)."""
    poprawne = votes.argmax(axis=1) == y
    if not poprawne.any():
        raise ValueError("żadna próbka kalibracyjna nie jest poprawnie sklasyfikowana")
    mg = margines(votes[poprawne], y[poprawne])
    return float(mg.mean()), float(mg.std())


def liczba_drzew(T, mu, sigma, k):
    """N* = T * (mu + k * sigma), zaokrąglone do najbliższej liczby całkowitej z [1, T]."""
    return int(np.clip(np.rint(T * (mu + k * sigma)), 1, T))


def spadek_marginesu(pred, y, l):
    """Kryterium MeanD-M (Yang i in., 2012) dla każdego drzewa: o ile średnio spada
    margines próbek, gdy to jedno drzewo usuniemy z zespołu. Średnia jest po próbkach,
    na które drzewo głosuje; drzewo bez głosów dostaje -inf.

    Skutek usunięcia drzewa zależy tylko od klasy, na którą głosowało, więc spadek
    liczymy raz dla każdej pary (próbka, klasa), a nie osobno dla każdego drzewa.
    """
    votes = glosy_z_predykcji(pred, l)
    mg = margines(votes, y)
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


def trafnosc_drzew(pred, y):
    """Trafność każdego drzewa na próbkach, na które głosuje (-inf, gdy na żadną)."""
    glosuje = pred >= 0
    ile = glosuje.sum(axis=1)
    trafne = ((pred == y) & glosuje).sum(axis=1)
    return np.where(ile > 0, trafne / np.maximum(ile, 1), -np.inf)


def ranking_drzew(pred, y, l, ranking="mean_d_m"):
    """Kolejność odwiedzania drzew (indeksy drzew, najważniejsze najpierw):

        "mean_d_m"  malejąco według średniego spadku marginesu po usunięciu drzewa
                    (MeanD-M, ranking użyty w eksperymentach artykułu),
        "accuracy"  malejąco według trafności pojedynczego drzewa,
        "none"      kolejność lasu (bez rankingu).

    Remisy rozstrzyga kolejność lasu.
    """
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
    """Wczesny stop oparty na marginesie: głosuje N* = T * (mu + k * sigma) pierwszych
    drzew rankingu, tyle samo dla każdej próbki.

        MarginEarlyStop(k=0.0)   k > 0: więcej drzew (dokładniej), k < 0: mniej (szybciej)
        ranking                  "mean_d_m" (artykuł), "accuracy" albo "none"
        n_calib                  ile próbek uczących użyć do kalibracji (None = wszystkie)

    fit(rf, X_train, y_train) wymaga danych, na których uczono las (bootstrap=True).
    Po fit: order (ranking drzew), mu, sigma, n_star (liczba głosujących drzew).
    set_k(k) zmienia k bez ponownej kalibracji (artykuł: strojenie k po wdrożeniu).
    """

    def __init__(self, k=0.0, ranking="mean_d_m", n_calib=20000, name=None):
        if ranking not in RANKINGS:
            raise ValueError(f"MarginEarlyStop: nieznany ranking {ranking!r}, dostępne: {RANKINGS}")
        self.k, self.ranking, self.n_calib = k, ranking, n_calib
        super().__init__(name or f"MARGIN_k{k:g}")

    def fit(self, rf, X_train=None, y_train=None):
        super().fit(rf)
        if X_train is None or y_train is None:
            raise ValueError(f"{self.name}: fit(rf, X_train, y_train) wymaga danych uczących "
                             f"lasu z etykietami (kalibracja marginesu)")
        pred, y = predykcje_kalibracyjne(rf, X_train, y_train, self.n_calib)
        self.order = ranking_drzew(pred, y, self.l, self.ranking)
        self.mu, self.sigma = statystyki_marginesu(glosy_z_predykcji(pred, self.l), y)
        return self.set_k(self.k)

    def set_k(self, k):
        """Ustawia k i wynikającą z niego liczbę głosujących drzew (nazwa metody zostaje)."""
        self.k = k
        self.n_star = liczba_drzew(self.T, self.mu, self.sigma, k)
        self.voters = [self.trees[i] for i in self.order[:self.n_star]]
        return self
