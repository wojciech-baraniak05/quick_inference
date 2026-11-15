"""Wspólny interfejs metod inferencji dla lasu losowego (opis: implementacja.md, rozdz. 1-3)."""
from collections.abc import Callable
from typing import Any, Literal, Self, get_args
from weakref import WeakKeyDictionary

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

Features = NDArray[np.float32]          # cechy w formacie drzew: float32, kształt (n, n_cech)
IntArray = NDArray[np.integer[Any]]     # głosy (n, l), indeksy klas, liczby drzew, progi
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Labels = NDArray[Any]                   # etykiety klas w typie rf.classes_
Prediction = tuple[Labels, IntArray]    # (y_pred, n_used) dla macierzy próbek
Order = Literal["fixed", "shuffled", "random"]
Multiclass = Literal["full", "pair"]    # tablica dla wszystkich klas | lider kontra druga klasa
Calib = Literal["oob", "val"]           # dane kalibracji: uczące out-of-bag | zbiór walidacyjny
Task = Literal["classification", "regression"]
Forest = RandomForestClassifier | RandomForestRegressor

# przyrostek nazwy metody; nieznaną kolejność odrzuca konstruktor (stąd .get w podklasach)
ORDER_SUFFIX: dict[Order, str] = {"fixed": "", "shuffled": "_shuf", "random": "_rand"}
CALIB_SUFFIX: dict[Calib, str] = {"oob": "", "val": "_val"}
MULTICLASS: tuple[Multiclass, ...] = get_args(Multiclass)
CALIBS: tuple[Calib, ...] = get_args(Calib)


def check_calib(cls_name: str, calib: str) -> None:
    if calib not in CALIBS:
        raise ValueError(f"{cls_name}: calib must be one of {CALIBS}, got {calib!r}")


def pi_tag(pi: float) -> str:
    """Poziom ufności w procentach do nazw plików i metod: 0.98 -> '98', 0.995 -> '99.5'."""
    return f"{round(pi * 100, 6):g}"


def to_float32(X: ArrayLike) -> Features:
    return np.ascontiguousarray(X, dtype=np.float32)


class FastTree:
    """Jedno drzewo lasu bez narzutu walidacji tree.predict."""

    __slots__ = ("apply", "leaf_class")
    apply: Callable[[Features], IntArray]       # próbki -> indeksy liści
    leaf_class: IntArray                        # indeks liścia -> indeks klasy

    def __init__(self, estimator: DecisionTreeClassifier) -> None:
        tree = estimator.tree_
        if tree.value.shape[1] != 1:
            raise ValueError("FastTree supports only single-output trees (n_outputs=1)")
        self.apply = tree.apply
        self.leaf_class = tree.value[:, 0, :].argmax(axis=1)

    def predict(self, X: Features) -> IntArray:
        return self.leaf_class[self.apply(X)]


class FastRegTree:
    """Jedno drzewo lasu regresyjnego: jak FastTree, ale liść daje wartość, a nie klasę."""

    __slots__ = ("apply", "leaf_value")
    apply: Callable[[Features], IntArray]       # próbki -> indeksy liści
    leaf_value: FloatArray                      # indeks liścia -> wyjście drzewa

    def __init__(self, estimator: DecisionTreeRegressor) -> None:
        tree = estimator.tree_
        if tree.value.shape[1] != 1:
            raise ValueError("FastRegTree supports only single-output trees (n_outputs=1)")
        self.apply = tree.apply
        self.leaf_value = np.ascontiguousarray(tree.value[:, 0, 0])

    def predict(self, X: Features) -> FloatArray:
        return self.leaf_value[self.apply(X)]


_FAST_TREES: WeakKeyDictionary[Forest, tuple[list[Any], list[Any]]] = WeakKeyDictionary()


def _fast_trees(rf: Forest, tree_cls: type[FastTree] | type[FastRegTree]) -> list[Any]:
    cached = _FAST_TREES.get(rf)
    if cached is None or cached[0] is not rf.estimators_:   # nowy las albo uczony ponownie
        cached = (rf.estimators_, [tree_cls(e) for e in rf.estimators_])
        _FAST_TREES[rf] = cached
    return cached[1]


def fast_trees(rf: RandomForestClassifier) -> list[FastTree]:
    """Lista FastTree w kolejności rf.estimators_, wspólna dla metod tego samego lasu."""
    return _fast_trees(rf, FastTree)


def fast_reg_trees(rf: RandomForestRegressor) -> list[FastRegTree]:
    """Lista FastRegTree w kolejności rf.estimators_, wspólna dla metod tego samego lasu."""
    return _fast_trees(rf, FastRegTree)


def stop_pair(votes: IntArray, threshold: IntArray) -> BoolArray:
    """Reguła dla pary lider - druga klasa: stop, gdy t_1 >= threshold[t_2]."""
    t = np.partition(votes, -2, axis=1)             # dwie ostatnie kolumny: t_2, t_1
    return t[:, -1] >= threshold[t[:, -2]]


def stop_pair_one(votes: IntArray, threshold: IntArray) -> bool:
    """stop_pair dla jednej próbki, votes ma kształt (l,)."""
    t = np.sort(votes)
    return bool(t[-1] >= threshold[t[-2]])


class InferenceMethod:
    """Bazowa klasa metod inferencji. Atrybuty bez wartości ustawia fit()."""

    name: str = "method"
    task: Task = "classification"
    pi: float | None = None     # poziom ufności; None dla metod, które go nie mają
    delta: float | None = None  # tolerancja w regresji; None dla metod, które jej nie mają
    rf: Any                     # RandomForestClassifier (w regresji RandomForestRegressor)
    trees: list[Any]            # FastTree (w regresji FastRegTree)
    classes: Labels
    T: int                      # liczba drzew
    l: int                      # liczba klas
    n_features: int

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name

    def fit(self, rf: RandomForestClassifier, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None, X_val: ArrayLike | None = None,
            y_val: ArrayLike | None = None) -> Self:
        """X_train, y_train: dane, na których uczono las; X_val, y_val: zbiór walidacyjny,
        którego las nie widział (dla metod z calib="val"). Metoda bierze z nich to, czego
        potrzebuje, reszta jest ignorowana."""
        if not hasattr(rf, "classes_"):
            raise ValueError(f"{self.name}: classification method, "
                             f"but the forest is {type(rf).__name__}")
        self.rf = rf
        self.trees = fast_trees(rf)
        self.classes = rf.classes_
        self.T = len(self.trees)
        self.l = len(self.classes)
        self.n_features = rf.n_features_in_
        return self

    def _X32(self, X: ArrayLike) -> Features:
        """Cechy w formacie dla self.trees. FastTree nie waliduje wejścia, więc kształt
        jest sprawdzany tutaj, raz na wywołanie predykcji."""
        X = to_float32(X)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"{self.name}: X has shape {X.shape}, "
                             f"expected (n, {self.n_features})")
        return X

    def predict_batch(self, X: ArrayLike) -> Prediction:
        raise NotImplementedError

    def predict_one(self, x: ArrayLike) -> tuple[Any, int]:
        """Predykcja jednej próbki o kształcie (1, n_cech): (etykieta, liczba odpytanych drzew)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class FixedVoting(InferenceMethod):
    """Hard voting stałej listy drzew self.voters, tej samej dla każdej próbki."""

    voters: list[FastTree]

    def fit(self, rf: RandomForestClassifier, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None, X_val: ArrayLike | None = None,
            y_val: ArrayLike | None = None) -> Self:
        super().fit(rf)
        self.voters = self.trees
        return self

    def predict_batch(self, X: ArrayLike) -> Prediction:
        X = self._X32(X)
        votes = np.zeros((len(X), self.l), dtype=int)
        rows = np.arange(len(X))
        for tree in self.voters:
            votes[rows, tree.predict(X)] += 1
        return self.classes[votes.argmax(axis=1)], np.full(len(X), len(self.voters))

    def predict_one(self, x: ArrayLike) -> tuple[Any, int]:
        x = self._X32(x)
        votes = np.zeros(self.l, dtype=int)
        for tree in self.voters:
            votes[tree.predict(x)[0]] += 1
        return self.classes[votes.argmax()], len(self.voters)


class SequentialPruning(InferenceMethod):
    """Wczesny stop: drzewa głosują po kolei, aż reguła stopu podklasy uzna wynik za pewny."""

    ORDERS: tuple[Order, ...] = get_args(Order)
    rng: np.random.Generator

    def __init__(self, name: str | None = None, order: Order = "fixed",
                 seed: int | None = None) -> None:
        if order not in self.ORDERS:
            raise ValueError(f"{type(self).__name__}: order must be one of {self.ORDERS}, "
                             f"got {order!r}")
        self.order, self.seed = order, seed
        super().__init__(name)

    def fit(self, rf: RandomForestClassifier, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None, X_val: ArrayLike | None = None,
            y_val: ArrayLike | None = None) -> Self:
        super().fit(rf)
        self.rng = np.random.default_rng(self.seed)
        if self.order == "shuffled":                # jedna permutacja lasu dla wszystkich próbek
            self.trees = [self.trees[j] for j in self.rng.permutation(self.T)]
        return self

    def _stop_batch(self, votes: IntArray) -> BoolArray:
        """Maska (n,) próbek, które mogą skończyć głosowanie; votes ma kształt (n, l)."""
        raise NotImplementedError

    def _stop_one(self, votes: IntArray) -> bool:
        """_stop_batch dla jednej próbki, votes ma kształt (l,). Podklasa może podać
        szybszą wersję skalarną."""
        return bool(self._stop_batch(votes[None, :])[0])

    def predict_batch(self, X: ArrayLike) -> Prediction:
        X = self._X32(X)
        if self.order == "random":
            return self._predict_batch_random(X)
        votes = np.zeros((len(X), self.l), dtype=int)
        n_used = np.full(len(X), self.T)
        active = np.arange(len(X))                  # próbki jeszcze niezdecydowane
        X_active = X                                # ich cechy, czyli X[active]
        for k, tree in enumerate(self.trees):
            votes[active, tree.predict(X_active)] += 1
            stop = self._stop_batch(votes[active])
            if stop.any():                          # cechy kopiujemy tylko, gdy ktoś odpadł
                n_used[active[stop]] = k + 1
                active = active[~stop]
                if len(active) == 0:
                    break
                X_active = X_active[~stop]
        return self.classes[votes.argmax(axis=1)], n_used

    def _predict_batch_random(self, X: Features) -> Prediction:
        n = len(X)
        votes = np.zeros((n, self.l), dtype=int)
        n_used = np.full(n, self.T)
        # perm[i, k] = drzewo, które próbka i pyta w kroku k
        perm = np.tile(np.arange(self.T, dtype=np.min_scalar_type(self.T)), (n, 1))
        perm = self.rng.permuted(perm, axis=1)
        active = np.arange(n)
        for k in range(self.T):
            # próbki pytające w tym kroku to samo drzewo idą do niego jednym wywołaniem
            tree_ids = perm[active, k]
            idx = np.argsort(tree_ids, kind="stable")
            samples, tree_ids = active[idx], tree_ids[idx]
            bounds = np.flatnonzero(np.diff(tree_ids)) + 1
            for group, j in zip(np.split(samples, bounds), tree_ids[np.r_[0, bounds]]):
                votes[group, self.trees[j].predict(X[group])] += 1
            stop = self._stop_batch(votes[active])
            n_used[active[stop]] = k + 1
            active = active[~stop]
            if len(active) == 0:
                break
        return self.classes[votes.argmax(axis=1)], n_used

    def predict_one(self, x: ArrayLike) -> tuple[Any, int]:
        x = self._X32(x)
        votes = np.zeros(self.l, dtype=int)
        trees = self.trees
        if self.order == "random":
            trees = [self.trees[j] for j in self.rng.permutation(self.T)]
        for k, tree in enumerate(trees):
            votes[tree.predict(x)[0]] += 1
            if self._stop_one(votes):
                break
        return self.classes[votes.argmax()], k + 1
