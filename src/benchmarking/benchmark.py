"""Benchmark metod jednego lasu na jednym zbiorze testowym (opis: implementacja.md, rozdz. 9)."""
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from methods.base import FloatArray, IntArray, Labels, Task

from .forest_methods import ForestMethods
from .modes import SUPPORTED_MODES, Mode, check_modes


class Benchmark:
    """Uruchamia metody z ForestMethods w wybranych scenariuszach i zbiera surowe wyniki:
    predykcje, liczby odpytanych drzew i czasy. Metryk nie liczy (to robi src/analysis)."""

    def __init__(self, models: ForestMethods, X_test: ArrayLike, y_test: ArrayLike,
                 metadata: Mapping[str, Any] | None = None, n_online: int = 500,
                 n_repeats: int = 1) -> None:
        self.models: ForestMethods | None = models
        self.X_test: NDArray[Any] | None = np.asarray(X_test)
        self.y_test: Labels = np.asarray(y_test)
        self.n_test = len(self.X_test)
        self.n_online = min(n_online, self.n_test)
        self.n_repeats = n_repeats
        self.T, self.classes = models.T, models.classes
        self.task: Task = models.task
        self.names: list[str] = []
        self.pi_levels: dict[str, float] = {}       # nazwa -> pi (tylko metody, które mają pi)
        self.deltas: dict[str, float] = {}          # nazwa -> tolerancja (regresja)
        self._snapshot_methods(models)
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.metadata.update(
            task=self.task, T=models.T, l=models.l, n_test=self.n_test, n_online=self.n_online,
            n_repeats=n_repeats, methods=self.names,
            date=datetime.now().isoformat(timespec="seconds"),
        )
        self.Y_pred: dict[Mode, Labels] = {}
        self.N_used: dict[Mode, IntArray] = {}
        self.times: dict[Mode, FloatArray] = {}

    def _snapshot_methods(self, models: ForestMethods) -> None:
        self.names = list(models.names)
        self.pi_levels = {m.name: m.pi for m in models.methods if m.pi is not None}
        self.deltas = {m.name: m.delta for m in models.methods if m.delta is not None}

    def _n(self, mode: Mode) -> int:
        return self.n_test if mode == "batch" else self.n_online

    def run(self, modes: Sequence[str] = SUPPORTED_MODES) -> Self:
        if self.models is None or self.X_test is None:
            raise RuntimeError("Benchmark has no forest or X_test after release(); "
                               "create a new Benchmark")
        self._snapshot_methods(self.models)         # metody mogły dojść po __init__
        self.metadata["methods"] = self.names
        if self.deltas:
            self.metadata["deltas"] = self.deltas
        for mode in check_modes(modes):
            Y, N, t = self.models.run(self.X_test[: self._n(mode)], mode=mode,
                                      n_repeats=self.n_repeats)
            self.Y_pred[mode], self.N_used[mode], self.times[mode] = Y, N, t
        return self

    def release(self) -> Self:
        """Zwalnia las, dane uczące i X_test. Wyniki zostają, run() przestaje działać."""
        self.models = None
        self.X_test = None
        return self

    def to_frame(self) -> pd.DataFrame:
        """Surowe wyniki w układzie pliku wyników (storage.py): wiersz na (scenariusz, metoda),
        kolumny to metadane przebiegu, mode, method, pi, delta, time_s i tablice y_true,
        y_pred, n_used."""
        # listy i słowniki metadanych (methods, deltas) są już wierszami i kolumnami pi, delta
        run = {k: v for k, v in self.metadata.items() if not isinstance(v, (list, dict))}
        rows = []
        for mode, Y in self.Y_pred.items():
            y_true = self.y_test[: Y.shape[1]]
            for i, name in enumerate(self.names):
                rows.append({**run, "mode": mode, "method": name,
                             "pi": self.pi_levels.get(name), "delta": self.deltas.get(name),
                             "time_s": float(self.times[mode][i]), "y_true": y_true,
                             "y_pred": Y[i], "n_used": self.N_used[mode][i]})
        return pd.DataFrame(rows)
