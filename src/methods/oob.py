"""Predykcje out-of-bag na danych uczących lasu (dla Hyper, MarginEarlyStop i metod własnych).
Opis: implementacja.md, rozdz. 8."""
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import FloatArray, Forest, IntArray, fast_reg_trees, fast_trees, to_float32


def sample_rows(rf: Forest, X_train: ArrayLike, n_max: int | None = None,
                  seed: int = 0) -> IntArray:
    """Indeksy (rosnąco) co najwyżej n_max losowych wierszy X_train; None = wszystkie."""
    n = len(np.asarray(X_train))
    if not rf.bootstrap:
        raise ValueError("out-of-bag votes need a forest with bootstrap=True")
    if getattr(rf, "_n_samples", n) != n:
        raise ValueError(f"X_train has {n} rows, but the forest was trained on {rf._n_samples}")
    if n_max is None or n_max >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, n_max, replace=False))


def oob_predictions(rf: Forest, X_train: ArrayLike,
                  rows: IntArray) -> Iterator[tuple[IntArray, NDArray[Any]]]:
    """Dla każdego drzewa (w kolejności rf.estimators_): pozycje w rows, których drzewo
    nie widziało przy uczeniu, oraz jego predykcje dla tych próbek (indeksy klas,
    a w lesie regresyjnym wartości)."""
    X_train = np.asarray(X_train)
    n = len(X_train)
    X = to_float32(X_train[rows])
    # indeksy bootstrapu każdego drzewa; generator sklearn nie trzyma ich naraz w pamięci
    if hasattr(rf, "_get_estimators_indices"):
        bootstrap_indices = rf._get_estimators_indices()
    else:
        bootstrap_indices = rf.estimators_samples_
    trees = fast_reg_trees(rf) if isinstance(rf, RandomForestRegressor) else fast_trees(rf)
    for tree, in_bag in zip(trees, bootstrap_indices):
        oob = np.flatnonzero(np.bincount(in_bag, minlength=n)[rows] == 0)
        yield oob, tree.predict(X[oob])


def oob_votes(rf: RandomForestClassifier, X_train: ArrayLike, n_prior: int | None = None,
              seed: int = 0) -> IntArray:
    """votes[n, i] = ile drzew, które nie widziały próbki n przy uczeniu, głosuje na klasę i."""
    rows = sample_rows(rf, X_train, n_prior, seed)
    votes = np.zeros((len(rows), len(rf.classes_)), dtype=int)
    for oob, pred in oob_predictions(rf, X_train, rows):
        votes[oob, pred] += 1
    return votes


def oob_variances(rf: RandomForestRegressor, X_train: ArrayLike, n_max: int | None = None,
                  seed: int = 0) -> FloatArray:
    """Dla próbek uczących z co najmniej dwoma drzewami out-of-bag: wariancja (z n - 1
    w mianowniku) wyjść drzew, które próbki nie widziały przy uczeniu."""
    rows = sample_rows(rf, X_train, n_max, seed)
    count = np.zeros(len(rows))
    mean = np.zeros(len(rows))
    S = np.zeros(len(rows))                         # suma kwadratów odchyleń od średniej
    for oob, y in oob_predictions(rf, X_train, rows):
        count[oob] += 1
        d = y - mean[oob]
        mean[oob] += d / count[oob]
        S[oob] += d * (y - mean[oob])
    ok = count >= 2
    return S[ok] / (count[ok] - 1)
