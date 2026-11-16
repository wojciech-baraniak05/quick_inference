"""Wspólne dane i lasy dla testów. Lasy są małe (51 drzew: nieparzyste T, jak zaleca
instrukcja.md, rozdz. 8.5, i tablica T51_L2_pi95.npz jest w repozytorium), żeby cały zestaw
liczył się w kilkanaście sekund.

Etykiety klas celowo nie są kolejnymi liczbami 0..l-1 (napisy, liczby z przerwami): metody
pracują na indeksach klas, a pomyłka indeks/etykieta to typowy błąd, który psuje wyniki
po cichu."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

T = 51


@dataclass
class Case:
    rf: Any
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _split(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                      random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _classification(n_classes: int, labels: np.ndarray, seed: int, n_samples: int = 900,
                    max_depth: int | None = 4) -> Case:
    X, y = make_classification(n_samples=n_samples, n_features=8, n_informative=5,
                               n_redundant=1, n_classes=n_classes, n_clusters_per_class=1,
                               class_sep=1.2, random_state=seed)
    y = labels[y]
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y, seed)
    rf = RandomForestClassifier(n_estimators=T, max_depth=max_depth, random_state=seed)
    rf.fit(X_train, y_train)
    return Case(rf, X_train, y_train, X_val, y_val, X_test, y_test)


@pytest.fixture(scope="session")
def binary() -> Case:
    """Dwie klasy o etykietach-napisach."""
    return _classification(2, np.array(["neg", "pos"]), seed=1)


@pytest.fixture(scope="session")
def three() -> Case:
    """Trzy klasy o etykietach 3, 7, 10."""
    return _classification(3, np.array([3, 7, 10]), seed=2)


@pytest.fixture(scope="session")
def five() -> Case:
    """Pięć klas: za dużo stanów dla pełnych tablic IB / HYPER przy 51 drzewach."""
    return _classification(5, np.array([-2, 0, 1, 5, 9]), seed=3, n_samples=1200)


@pytest.fixture(scope="session")
def regression() -> Case:
    X, y = make_regression(n_samples=700, n_features=6, n_informative=4, noise=10.0,
                           random_state=4)
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y, 4)
    rf = RandomForestRegressor(n_estimators=T, max_depth=5, random_state=4)
    rf.fit(X_train, y_train)
    return Case(rf, X_train, y_train, X_val, y_val, X_test, y_test)


def all_vote_states(T: int, l: int, s_min: int = 1, s_max: int | None = None) -> np.ndarray:
    """Wszystkie wektory głosów (n, l) o sumie w [s_min, s_max] (domyślnie do T - 1)."""
    s_max = T - 1 if s_max is None else s_max
    grid = np.indices((s_max + 1,) * l).reshape(l, -1).T
    s = grid.sum(axis=1)
    return grid[(s >= s_min) & (s <= s_max)]


def tree_index_predictions(rf, X) -> np.ndarray:
    """(T, n): indeks klasy każdego drzewa lasu, liczony przez scikit-learn (niezależnie od
    FastTree)."""
    return np.stack([e.predict(np.asarray(X, dtype=np.float32)).astype(int)
                     for e in rf.estimators_])


def full_vote(rf, X, trees: np.ndarray | None = None) -> np.ndarray:
    """Hard voting wybranych drzew (domyślnie wszystkich), remis -> niższy indeks klasy."""
    pred = tree_index_predictions(rf, X)
    if trees is not None:
        pred = pred[trees]
    l = len(rf.classes_)
    votes = np.stack([(pred == c).sum(axis=0) for c in range(l)], axis=1)
    return rf.classes_[votes.argmax(axis=1)]


def check_batch_equals_online(method, X) -> tuple[np.ndarray, np.ndarray]:
    """Batch i online muszą dać te same predykcje i liczby drzew (kolejność inna niż random)."""
    y_b, n_b = method.predict_batch(X)
    online = [method.predict_one(x[None, :]) for x in np.asarray(X)]
    y_o = np.array([y for y, _ in online])
    n_o = np.array([n for _, n in online])
    np.testing.assert_array_equal(n_b, n_o)
    if np.issubdtype(np.asarray(y_b).dtype, np.floating):
        np.testing.assert_allclose(y_b, y_o, rtol=1e-12, atol=1e-12)
    else:
        np.testing.assert_array_equal(y_b, y_o)
    return y_b, n_b
