"""Predykcje lasu na zbiorze walidacyjnym: zamiennik predykcji out-of-bag (oob.py) dla metod
z calib="val" (HYPER, metody własne, MARGIN). Zbioru walidacyjnego las nie widział przy
uczeniu, więc na każdą próbkę głosują wszystkie drzewa. Opis: implementacja.md, rozdz. 8."""
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier

from .base import Features, IntArray, fast_trees, to_float32


def val_rows(X_val: ArrayLike, n_max: int | None = None, seed: int = 0) -> IntArray:
    """Indeksy (rosnąco) co najwyżej n_max losowych wierszy X_val; None = wszystkie."""
    n = len(np.asarray(X_val))
    if n == 0:
        raise ValueError("the validation set is empty")
    if n_max is None or n_max >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, n_max, replace=False))


def _features(rf: RandomForestClassifier, X_val: ArrayLike, rows: IntArray) -> Features:
    # FastTree nie sprawdza wejścia, a X_val (inaczej niż X_train w oob.py) nie musi
    # pochodzić z danych lasu
    X = to_float32(np.asarray(X_val)[rows])
    if X.ndim != 2 or X.shape[1] != rf.n_features_in_:
        raise ValueError(f"X_val has shape {X.shape}, expected (n, {rf.n_features_in_})")
    return X


def val_votes(rf: RandomForestClassifier, X_val: ArrayLike, n_max: int | None = None,
              seed: int = 0) -> IntArray:
    """votes[n, i] = ile drzew głosuje na klasę i dla próbki walidacyjnej n (w wierszu
    suma T)."""
    rows = val_rows(X_val, n_max, seed)
    X = _features(rf, X_val, rows)
    votes = np.zeros((len(rows), len(rf.classes_)), dtype=int)
    r = np.arange(len(rows))
    for tree in fast_trees(rf):
        votes[r, tree.predict(X)] += 1
    return votes


def val_predictions(rf: RandomForestClassifier, X_val: ArrayLike, y_val: ArrayLike,
                    n_max: int | None = None, seed: int = 0) -> tuple[NDArray[Any], IntArray]:
    """(pred, y): pred[i, n] to indeks klasy drzewa i (kolejność rf.estimators_) dla próbki
    walidacyjnej n, y[n] to indeks jej klasy prawdziwej w rf.classes_. Próbki klas, których
    las nie zna (nie było ich w danych uczących), są pomijane."""
    X_val, y_val = np.asarray(X_val), np.asarray(y_val)
    if len(y_val) != len(X_val):
        raise ValueError(f"X_val has {len(X_val)} rows, y_val {len(y_val)}")
    rows = val_rows(X_val, n_max, seed)
    rows = rows[np.isin(y_val[rows], rf.classes_)]
    X = _features(rf, X_val, rows)
    trees = fast_trees(rf)
    pred = np.empty((len(trees), len(rows)), dtype=np.int16)
    for i, tree in enumerate(trees):
        pred[i] = tree.predict(X)
    return pred, np.searchsorted(rf.classes_, y_val[rows])
