"""Predykcje out-of-bag na danych uczących lasu (dla Hyper, MarginEarlyStop i metod własnych).
Opis: implementacja.md, rozdz. 8."""
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import FloatArray, Forest, IntArray, fast_reg_trees, fast_trees, to_float32


def probka_uczaca(rf: Forest, X_train: ArrayLike, n_max: int | None = None,
                  seed: int = 0) -> IntArray:
    """Indeksy (rosnąco) co najwyżej n_max losowych wierszy X_train; None = wszystkie."""
    n = len(np.asarray(X_train))
    if not rf.bootstrap:
        raise ValueError("głosy out-of-bag wymagają lasu z bootstrap=True")
    if getattr(rf, "_n_samples", n) != n:
        raise ValueError(f"X_train ma {n} wierszy, a las uczono na {rf._n_samples}")
    if n_max is None or n_max >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, n_max, replace=False))


def predykcje_oob(rf: Forest, X_train: ArrayLike,
                  wiersze: IntArray) -> Iterator[tuple[IntArray, NDArray[Any]]]:
    """Dla każdego drzewa (w kolejności rf.estimators_): pozycje w wiersze, których drzewo
    nie widziało przy uczeniu, oraz jego predykcje dla tych próbek (indeksy klas,
    a w lesie regresyjnym wartości)."""
    X_train = np.asarray(X_train)
    n = len(X_train)
    X = to_float32(X_train[wiersze])
    # indeksy bootstrapu każdego drzewa; generator sklearn nie trzyma ich naraz w pamięci
    if hasattr(rf, "_get_estimators_indices"):
        indeksy = rf._get_estimators_indices()
    else:
        indeksy = rf.estimators_samples_
    trees = fast_reg_trees(rf) if isinstance(rf, RandomForestRegressor) else fast_trees(rf)
    for tree, w_probie in zip(trees, indeksy):
        oob = np.flatnonzero(np.bincount(w_probie, minlength=n)[wiersze] == 0)
        yield oob, tree.predict(X[oob])


def glosy_oob(rf: RandomForestClassifier, X_train: ArrayLike, n_prior: int | None = None,
              seed: int = 0) -> IntArray:
    """votes[n, i] = ile drzew, które nie widziały próbki n przy uczeniu, głosuje na klasę i."""
    wiersze = probka_uczaca(rf, X_train, n_prior, seed)
    votes = np.zeros((len(wiersze), len(rf.classes_)), dtype=int)
    for oob, pred in predykcje_oob(rf, X_train, wiersze):
        votes[oob, pred] += 1
    return votes


def wariancje_oob(rf: RandomForestRegressor, X_train: ArrayLike, n_max: int | None = None,
                  seed: int = 0) -> FloatArray:
    """Dla próbek uczących z co najmniej dwoma drzewami out-of-bag: wariancja (z n - 1
    w mianowniku) wyjść drzew, które próbki nie widziały przy uczeniu."""
    wiersze = probka_uczaca(rf, X_train, n_max, seed)
    ile = np.zeros(len(wiersze))
    srednia = np.zeros(len(wiersze))
    S = np.zeros(len(wiersze))                      # suma kwadratów odchyleń od średniej
    for oob, y in predykcje_oob(rf, X_train, wiersze):
        ile[oob] += 1
        d = y - srednia[oob]
        srednia[oob] += d / ile[oob]
        S[oob] += d * (y - srednia[oob])
    ok = ile >= 2
    return S[ok] / (ile[ok] - 1)
