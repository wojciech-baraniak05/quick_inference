"""Zbiór metod inferencji dopasowanych do jednego lasu (opis: implementacja.md, rozdz. 9)."""
import time
from collections.abc import Iterable
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier

from methods import FullVoting, InferenceMethod, SklearnRF
from methods.base import FloatArray, IntArray, Labels, Prediction

from .modes import Mode, check_modes

RunResult = tuple[Labels, IntArray, FloatArray]     # (Y_pred, N_used, times), wiersz = metoda


class methods_rf:
    """Metody inferencji dla jednego wytrenowanego lasu; run() zbiera ich wyniki w macierze."""

    def __init__(self, rf: RandomForestClassifier,
                 methods: Iterable[InferenceMethod] | None = None,
                 X_train: ArrayLike | None = None, y_train: ArrayLike | None = None) -> None:
        self.rf = rf
        self.X_train, self.y_train = X_train, y_train       # przekazywane do fit() każdej metody
        self.classes: Labels = rf.classes_
        self.T = len(rf.estimators_)
        self.l = len(rf.classes_)
        self.methods: list[InferenceMethod] = []
        for m in (methods if methods is not None else [SklearnRF(), FullVoting()]):
            self.add_method(m)
        self.Y_pred: dict[Mode, Labels] = {}
        self.N_used: dict[Mode, IntArray] = {}
        self.times: dict[Mode, FloatArray] = {}

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.methods]

    def add_method(self, method: InferenceMethod) -> Self:
        if method.name in self.names:
            raise ValueError(f"metoda o nazwie '{method.name}' już istnieje")
        self.methods.append(method.fit(self.rf, self.X_train, self.y_train))
        return self

    def index(self, name: str) -> int:
        return self.names.index(name)

    def _run_method(self, method: InferenceMethod, X: NDArray[Any], mode: Mode) -> Prediction:
        if mode == "batch":
            return method.predict_batch(X)
        wyniki = [method.predict_one(x[None, :]) for x in X]
        y, n = zip(*wyniki)
        return np.array(y), np.array(n)

    def run(self, X: NDArray[Any], mode: Mode = "batch", n_repeats: int = 1) -> RunResult:
        check_modes([mode])
        Y = np.empty((len(self.methods), len(X)), dtype=self.classes.dtype)
        N = np.empty((len(self.methods), len(X)), dtype=int)
        times = np.empty(len(self.methods))
        for i, method in enumerate(self.methods):
            pomiary = []
            for _ in range(n_repeats):
                start = time.perf_counter()
                y, n = self._run_method(method, X, mode)
                pomiary.append(time.perf_counter() - start)
            Y[i], N[i], times[i] = y, n, np.median(pomiary)
        self.Y_pred[mode], self.N_used[mode], self.times[mode] = Y, N, times
        return Y, N, times
