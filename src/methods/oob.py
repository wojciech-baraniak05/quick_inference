"""Predykcje out-of-bag na danych uczących lasu (dla Hyper i MarginEarlyStop).
Opis: implementacja.md, rozdz. 8."""
from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier

from .base import IntArray, fast_trees, to_float32


def probka_uczaca(rf: RandomForestClassifier, X_train: ArrayLike, n_max: int | None = None,
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


def predykcje_oob(rf: RandomForestClassifier, X_train: ArrayLike,
                  wiersze: IntArray) -> Iterator[tuple[IntArray, IntArray]]:
    """Dla każdego drzewa (w kolejności rf.estimators_): pozycje w wiersze, których drzewo
    nie widziało przy uczeniu, oraz indeksy klas przewidziane dla tych próbek."""
    X_train = np.asarray(X_train)
    n = len(X_train)
    X = to_float32(X_train[wiersze])
    # indeksy bootstrapu każdego drzewa; generator sklearn nie trzyma ich naraz w pamięci
    if hasattr(rf, "_get_estimators_indices"):
        indeksy = rf._get_estimators_indices()
    else:
        indeksy = rf.estimators_samples_
    for tree, w_probie in zip(fast_trees(rf), indeksy):
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
