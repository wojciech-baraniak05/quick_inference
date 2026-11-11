"""Wyniki wielu benchmarków (wiele zbiorów danych i powtórzeń)."""
from pathlib import Path

import pandas as pd

from .modes import SUPPORTED_MODES


class benchmark_suite:
    """Zbiór wyników benchmark dla wielu modeli (np. wiele zbiorów danych
    i kilka powtórzeń z różnymi seedami).

    Każdy benchmark jest identyfikowany wartościami kluczy z jego metadanych
    (domyślnie ('dataset', 'seed')). Ostatni klucz to powtórzenie, po którym
    uśrednia aggregate()."""

    SUMMARY_COLS = ["accuracy", "f1", "agree_rf", "agree_full", "ms_per_sample",
                    "speedup_vs_rf", "speedup_vs_full", "trees_mean", "trees_frac"]

    def __init__(self, keys=("dataset", "seed")):
        self.keys = list(keys)
        self.runs = []

    def add(self, b):
        brak = [k for k in self.keys if k not in b.metadata]
        if brak:
            raise ValueError(f"w metadanych benchmarku brakuje kluczy: {brak}")
        self.runs.append(b)
        return self

    def __len__(self):
        return len(self.runs)

    @property
    def modes(self):
        """Scenariusze, które mają wyniki we wszystkich benchmarkach."""
        return [m for m in SUPPORTED_MODES if self.runs and all(m in b.Y_pred for b in self.runs)]

    def results(self, mode="batch"):
        """Wyniki wszystkich benchmarków: jeden wiersz na (klucze..., metoda)."""
        frames = []
        for b in self.runs:
            r = b.results(mode).reset_index()
            for k in reversed(self.keys):
                r.insert(0, k, b.metadata[k])
            frames.append(r)
        return pd.concat(frames, ignore_index=True).set_index([*self.keys, "method"])

    def aggregate(self, mode="batch", stats=("mean", "std"), overall=False):
        """Statystyki po powtórzeniach (ostatni klucz) dla każdego zbioru i metody.
        overall=True uśrednia dodatkowo po wszystkich zbiorach (wiersz na metodę)."""
        df = self.results(mode).reset_index()
        group = ["method"] if overall else [*self.keys[:-1], "method"]
        df = df.drop(columns=[k for k in self.keys if k not in group])
        stats = stats if isinstance(stats, str) else list(stats)
        return df.groupby(group, sort=False).agg(stats)

    def summary(self, cols=None):
        cols = cols or self.SUMMARY_COLS
        with pd.option_context("display.width", 250, "display.max_columns", None,
                               "display.max_rows", None,
                               "display.float_format", "{:.4f}".format):
            for mode in self.modes:
                per_zbior = self.aggregate(mode, stats="mean")
                kol = [c for c in cols if c in per_zbior.columns]
                print(f"\n[{mode}] średnia z powtórzeń ({self.keys[-1]}) dla każdego zbioru:")
                print(per_zbior[kol])
                print(f"\n[{mode}] średnia po wszystkich zbiorach i powtórzeniach:")
                print(self.aggregate(mode, stats="mean", overall=True)[kol])

    def save(self, out_dir):
        """Zapisuje macierze każdego benchmarku (.npz) oraz tabele wyników (.csv)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for b in self.runs:
            nazwa = "_".join(f"{k}{b.metadata[k]}" if k != "dataset" else str(b.metadata[k])
                             for k in self.keys)
            b.save(out_dir / "runs" / f"{nazwa}.npz")
        for mode in self.modes:
            self.results(mode).to_csv(out_dir / f"results_{mode}.csv")
            self.aggregate(mode).to_csv(out_dir / f"aggregate_{mode}.csv")
        return out_dir
