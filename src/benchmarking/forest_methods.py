"""Zbiór metod inferencji dopasowanych do jednego lasu (opis: implementacja.md, rozdz. 9)."""
import time
from collections.abc import Iterable
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestRegressor

from methods import FullAveraging, FullVoting, InferenceMethod, SklearnRF, SklearnRFReg
from methods.base import FloatArray, Forest, IntArray, Labels, Prediction, Task

from .modes import Mode, check_modes

RunResult = tuple[Labels, IntArray, FloatArray]     # (Y_pred, N_used, times), wiersz = metoda


class ForestMethods:
    """Metody inferencji dla jednego wytrenowanego lasu; run() zbiera ich wyniki w macierze."""

    def __init__(self, rf: Forest,
                 methods: Iterable[InferenceMethod] | None = None,
                 X_train: ArrayLike | None = None, y_train: ArrayLike | None = None,
                 X_val: ArrayLike | None = None, y_val: ArrayLike | None = None) -> None:
        self.rf = rf
        # przekazywane do fit() każdej metody; X_val, y_val: zbiór walidacyjny, którego las
        # nie widział (metody z calib="val")
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        is_regression = isinstance(rf, RandomForestRegressor)
        self.task: Task = "regression" if is_regression else "classification"
        self.classes: Labels = np.empty(0) if is_regression else rf.classes_  # regresja: bez klas
        self.T = len(rf.estimators_)
        self.l = len(self.classes)
        self.methods: list[InferenceMethod] = []
        if methods is None:
            methods = ([SklearnRFReg(), FullAveraging()] if is_regression
                       else [SklearnRF(), FullVoting()])
        for m in methods:
            self.add_method(m)
        self.Y_pred: dict[Mode, Labels] = {}
        self.N_used: dict[Mode, IntArray] = {}
        self.times: dict[Mode, FloatArray] = {}

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.methods]

    def add_method(self, method: InferenceMethod) -> Self:
        if method.name in self.names:
            raise ValueError(f"method name '{method.name}' is already used")
        self.methods.append(method.fit(self.rf, self.X_train, self.y_train,
                                       self.X_val, self.y_val))
        return self

    def index(self, name: str) -> int:
        return self.names.index(name)

    def _run_method(self, method: InferenceMethod, X: NDArray[Any], mode: Mode) -> Prediction:
        if mode == "batch":
            return method.predict_batch(X)
        preds = [method.predict_one(x[None, :]) for x in X]
        y, n = zip(*preds)
        return np.array(y), np.array(n)

    def run(self, X: NDArray[Any], mode: Mode = "batch", n_repeats: int = 1) -> RunResult:
        check_modes([mode])
        Y = np.empty((len(self.methods), len(X)), dtype=self.classes.dtype)  # regresja: float64
        N = np.empty((len(self.methods), len(X)), dtype=int)
        times = np.empty(len(self.methods))
        for i, method in enumerate(self.methods):
            timings = []
            for _ in range(n_repeats):
                start = time.perf_counter()
                y, n = self._run_method(method, X, mode)
                timings.append(time.perf_counter() - start)
            Y[i], N[i], times[i] = y, n, np.median(timings)
        self.Y_pred[mode], self.N_used[mode], self.times[mode] = Y, N, times
        return Y, N, times
