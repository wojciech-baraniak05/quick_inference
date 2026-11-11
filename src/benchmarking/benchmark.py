"""Benchmark metod jednego lasu na jednym zbiorze testowym."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from methods import FullVoting, SklearnRF

from .modes import SUPPORTED_MODES, check_modes


class benchmark:
    """Uruchamia wszystkie metody z methods_rf w wybranych scenariuszach
    (patrz benchmarking.modes) i porównuje je metrykami jakości, zgodnością
    z rf_sklearn / full_voting oraz czasem predykcji.

    Kolumny porównania z metodą odniesienia (agree_*, speedup_vs_*) są w wynikach
    tylko wtedy, gdy ta metoda jest w zbiorze (np. bez rf_sklearn, gdy
    sklearn_soft_voting = false)."""

    REF_RF = SklearnRF.name
    REF_FULL = FullVoting.name

    def __init__(self, models, X_test, y_test, metadata=None, n_online=500, n_repeats=1):
        self.models = models
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test)
        self.n_online = min(n_online, len(self.X_test))
        self.n_repeats = n_repeats
        self.metadata = dict(metadata or {})
        self.metadata.update(
            T=models.T, l=models.l, n_test=len(self.X_test), n_online=self.n_online,
            n_repeats=n_repeats, methods=models.names,
            date=datetime.now().isoformat(timespec="seconds"),
        )
        self.Y_pred, self.N_used, self.times = {}, {}, {}

    def _X_y(self, mode):
        if mode == "batch":
            return self.X_test, self.y_test
        return self.X_test[: self.n_online], self.y_test[: self.n_online]

    def run(self, modes=SUPPORTED_MODES):
        for mode in check_modes(modes):
            X, _ = self._X_y(mode)
            Y, N, t = self.models.run(X, mode=mode, n_repeats=self.n_repeats)
            self.Y_pred[mode], self.N_used[mode], self.times[mode] = Y, N, t
        return self

    def _ref(self, name):
        """Indeks metody odniesienia (albo None, jeśli jej nie ma)."""
        names = self.models.names
        return names.index(name) if name in names else None

    def results(self, mode="batch"):
        if mode not in self.Y_pred:
            raise RuntimeError(f"brak wyników dla '{mode}', najpierw wywołaj run()")
        Y, N, times = self.Y_pred[mode], self.N_used[mode], self.times[mode]
        _, y_true = self._X_y(mode)
        refs = {"rf": self._ref(self.REF_RF), "full": self._ref(self.REF_FULL)}
        refs = {k: i for k, i in refs.items() if i is not None}
        n = Y.shape[1]

        rows = []
        for i, name in enumerate(self.models.names):
            y = Y[i]
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
                trees_frac=N[i].mean() / self.models.T,
            )
            rows.append(row)
        return pd.DataFrame(rows).set_index("method")

    def summary(self):
        print("metadane:")
        for k, v in self.metadata.items():
            print(f"  {k}: {v}")
        with pd.option_context("display.width", 200, "display.max_columns", None,
                               "display.float_format", "{:.4f}".format):
            for mode in self.Y_pred:
                n = self.Y_pred[mode].shape[1]
                print(f"\n[{mode}] ({n} próbek)")
                print(self.results(mode))

    def to_dict(self):
        d = {"metadata": json.dumps(self.metadata, default=str),
             "names": np.array(self.models.names),
             "classes": self.models.classes,
             "y_test": self.y_test}
        for mode in self.Y_pred:
            d[f"Y_pred_{mode}"] = self.Y_pred[mode]
            d[f"N_used_{mode}"] = self.N_used[mode]
            d[f"times_{mode}"] = self.times[mode]
        return d

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_dict())
        return path
