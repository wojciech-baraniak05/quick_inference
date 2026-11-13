"""Benchmark metod jednego lasu na jednym zbiorze testowym (opis: implementacja.md, rozdz. 9)."""
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, root_mean_squared_error)

from methods import FullAveraging, FullVoting, SklearnRF
from methods.base import FloatArray, IntArray, Labels, Task

from .forest_methods import methods_rf
from .modes import SUPPORTED_MODES, Mode, check_modes


class benchmark:
    """Uruchamia metody z methods_rf w wybranych scenariuszach i porównuje je jakością,
    zgodnością z metodami odniesienia oraz czasem predykcji."""

    REF_RF = SklearnRF.name
    REF_FULL = FullVoting.name                      # w regresji: REF_FULL_REG
    REF_FULL_REG = FullAveraging.name
    REFS = (REF_RF, REF_FULL, REF_FULL_REG)

    def __init__(self, models: methods_rf, X_test: ArrayLike, y_test: ArrayLike,
                 metadata: Mapping[str, Any] | None = None, n_online: int = 500,
                 n_repeats: int = 1) -> None:
        self.models: methods_rf | None = models
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
        self._results: dict[Mode, pd.DataFrame] = {}

    def _snapshot_methods(self, models: methods_rf) -> None:
        self.names = list(models.names)
        self.pi_levels = {m.name: m.pi for m in models.methods if m.pi is not None}
        self.deltas = {m.name: m.delta for m in models.methods if m.delta is not None}

    def _n(self, mode: Mode) -> int:
        return self.n_test if mode == "batch" else self.n_online

    def run(self, modes: Sequence[str] = SUPPORTED_MODES) -> Self:
        if self.models is None or self.X_test is None:
            raise RuntimeError("benchmark po release() nie ma już lasu ani X_test; "
                               "utwórz nowy benchmark")
        self._snapshot_methods(self.models)         # metody mogły dojść po __init__
        self.metadata["methods"] = self.names
        if self.deltas:
            self.metadata["deltas"] = self.deltas
        for mode in check_modes(modes):
            Y, N, t = self.models.run(self.X_test[: self._n(mode)], mode=mode,
                                      n_repeats=self.n_repeats)
            self.Y_pred[mode], self.N_used[mode], self.times[mode] = Y, N, t
            self._results.pop(mode, None)
        return self

    def release(self) -> Self:
        """Zwalnia las, dane uczące i X_test. Wyniki zostają, run() przestaje działać."""
        self.models = None
        self.X_test = None
        return self

    def _ref(self, name: str) -> int | None:
        return self.names.index(name) if name in self.names else None

    def results(self, mode: Mode = "batch") -> pd.DataFrame:
        """Tabela wyników, wiersz na metodę; liczona raz na scenariusz."""
        if mode not in self.Y_pred:
            raise RuntimeError(f"brak wyników dla '{mode}', najpierw wywołaj run()")
        if mode not in self._results:
            self._results[mode] = self._compute_results(mode)
        return self._results[mode].copy()

    def _compute_results(self, mode: Mode) -> pd.DataFrame:
        Y, N, times = self.Y_pred[mode], self.N_used[mode], self.times[mode]
        y_true = self.y_test[: self._n(mode)]
        # kolumny agree_* i speedup_vs_* są tylko dla metod odniesienia obecnych w zbiorze
        regresja = self.task == "regression"
        full = self.REF_FULL_REG if regresja else self.REF_FULL
        refs = {k: i for k, i in (("rf", self._ref(self.REF_RF)),
                                  ("full", self._ref(full))) if i is not None}
        n = Y.shape[1]

        rows = []
        for i, name in enumerate(self.names):
            y = Y[i]
            if regresja:
                row = {
                    "method": name,
                    "mae": mean_absolute_error(y_true, y),
                    "rmse": root_mean_squared_error(y_true, y),
                    "r2": r2_score(y_true, y),
                }
                for k, j in refs.items():
                    # zgodność: różnica względem odniesienia w tolerancji metody; metody
                    # bez tolerancji muszą dać ten sam wynik (z dokładnością arytmetyki)
                    roznica = np.abs(y - Y[j])
                    zgodne = (roznica <= self.deltas[name] if name in self.deltas
                              else np.isclose(y, Y[j]))
                    row[f"agree_{k}"] = zgodne.mean()
                    row[f"mad_{k}"] = roznica.mean()
                    row[f"maxdiff_{k}"] = roznica.max()
            else:
                row = {
                    "method": name,
                    "accuracy": accuracy_score(y_true, y),
                    "precision": precision_score(y_true, y, average="macro", zero_division=0),
                    "recall": recall_score(y_true, y, average="macro", zero_division=0),
                    "f1": f1_score(y_true, y, average="macro", zero_division=0),
                }
                for k, j in refs.items():
                    row[f"agree_{k}"] = (y == Y[j]).mean()
            row["time_s"] = times[i]
            row["ms_per_sample"] = times[i] / n * 1e3
            for k, j in refs.items():
                row[f"speedup_vs_{k}"] = times[j] / times[i]
            row.update(
                trees_mean=N[i].mean(),
                trees_median=np.median(N[i]),
                trees_max=N[i].max(),
                trees_frac=N[i].mean() / self.T,
            )
            rows.append(row)
        return pd.DataFrame(rows).set_index("method")

    def summary(self) -> None:
        print("metadane:")
        for k, v in self.metadata.items():
            print(f"  {k}: {v}")
        with pd.option_context("display.width", 200, "display.max_columns", None,
                               "display.float_format", "{:.4f}".format):
            for mode in self.Y_pred:
                n = self.Y_pred[mode].shape[1]
                print(f"\n[{mode}] ({n} próbek)")
                print(self.results(mode))

    def to_dict(self) -> dict[str, Any]:
        d = {"metadata": json.dumps(self.metadata, default=str),
             "names": np.array(self.names),
             "classes": self.classes,
             "y_test": self.y_test}
        for mode in self.Y_pred:
            d[f"Y_pred_{mode}"] = self.Y_pred[mode]
            d[f"N_used_{mode}"] = self.N_used[mode]
            d[f"times_{mode}"] = self.times[mode]
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_dict())
        return path
