"""Wyniki wielu benchmarków: wiele zbiorów danych i powtórzeń (opis: implementacja.md, rozdz. 9)."""
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import pandas as pd

from .benchmark import Benchmark
from .modes import SUPPORTED_MODES, Mode


class BenchmarkSuite:
    """Benchmarki identyfikowane kluczami z metadanych; ostatni klucz to powtórzenie,
    po którym uśrednia aggregate()."""

    # kolumny jakości zależą od zadania; summary() pokazuje te, które są w wynikach
    SUMMARY_COLS = ["accuracy", "f1", "mae", "rmse", "r2", "agree_rf", "agree_full", "mad_full",
                    "ms_per_sample", "speedup_vs_rf", "speedup_vs_full", "trees_mean", "trees_frac"]

    def __init__(self, keys: Sequence[str] = ("dataset", "seed"),
                 out_dir: str | Path | None = None) -> None:
        self.keys = list(keys)
        self.out_dir = None if out_dir is None else Path(out_dir)
        self.runs: list[Benchmark] = []

    def add(self, b: Benchmark) -> Self:
        """Dodaje benchmark; z out_dir od razu zapisuje jego .npz i aktualne tabele .csv."""
        missing = [k for k in self.keys if k not in b.metadata]
        if missing:
            raise ValueError(f"benchmark metadata is missing keys: {missing}")
        self.runs.append(b)
        if self.out_dir is not None:
            b.save(self._run_path(self.out_dir, b))
            try:
                self._save_tables(self.out_dir)
            except OSError as e:
                # np. plik .csv otwarty w Excelu; wyniki przebiegu są już w .npz,
                # a tabele zostaną zapisane przy następnym add() albo w save()
                print(f"  result tables not written ({e}); retried on next add()/save()")
        return self

    def __len__(self) -> int:
        return len(self.runs)

    @property
    def modes(self) -> list[Mode]:
        """Scenariusze, które mają wyniki we wszystkich benchmarkach."""
        return [m for m in SUPPORTED_MODES if self.runs and all(m in b.Y_pred for b in self.runs)]

    def results(self, mode: Mode = "batch") -> pd.DataFrame:
        """Jeden wiersz na (klucze..., metoda)."""
        frames = []
        for b in self.runs:
            r = b.results(mode).reset_index()
            for k in reversed(self.keys):
                r.insert(0, k, b.metadata[k])
            frames.append(r)
        return pd.concat(frames, ignore_index=True).set_index([*self.keys, "method"])

    def aggregate(self, mode: Mode = "batch", stats: str | Sequence[str] = ("mean", "std"),
                  overall: bool = False) -> pd.DataFrame:
        """Statystyki po powtórzeniach dla każdego zbioru i metody; overall=True uśrednia
        dodatkowo po zbiorach (wiersz na metodę)."""
        df = self.results(mode).reset_index()
        group = ["method"] if overall else [*self.keys[:-1], "method"]
        df = df.drop(columns=[k for k in self.keys if k not in group])
        stats = stats if isinstance(stats, str) else list(stats)
        return df.groupby(group, sort=False).agg(stats)

    def summary(self, cols: Sequence[str] | None = None) -> None:
        cols = cols or self.SUMMARY_COLS
        with pd.option_context("display.width", 250, "display.max_columns", None,
                               "display.max_rows", None,
                               "display.float_format", "{:.4f}".format):
            rep = self.keys[-1]
            n_rep = len({b.metadata[rep] for b in self.runs})
            groups = ", ".join(self.keys[:-1]) or "run"
            for mode in self.modes:
                per_dataset = self.aggregate(mode, stats="mean")
                shown = [c for c in cols if c in per_dataset.columns]
                print(f"\n[{mode}] mean over {n_rep} {rep}s, per {groups}:")
                print(per_dataset[shown])
                print(f"\n[{mode}] mean over all {len(self.runs)} runs:")
                print(self.aggregate(mode, stats="mean", overall=True)[shown])

    def _run_path(self, out_dir: Path, b: Benchmark) -> Path:
        name = "_".join(f"{k}{b.metadata[k]}" if k != "dataset" else str(b.metadata[k])
                         for k in self.keys)
        return out_dir / "runs" / f"{name}.npz"

    def _save_tables(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for mode in self.modes:
            for name, df in ((f"results_{mode}.csv", self.results(mode)),
                              (f"aggregate_{mode}.csv", self.aggregate(mode))):
                # zapis przez plik tymczasowy: przerwanie nie zostawia uciętej tabeli
                tmp = out_dir / f"{name}.tmp"
                df.to_csv(tmp)
                tmp.replace(out_dir / name)

    def save(self, out_dir: str | Path | None = None) -> Path:
        """Zapisuje .npz każdego benchmarku i tabele .csv; domyślnie do out_dir z konstruktora."""
        target = self.out_dir if out_dir is None else Path(out_dir)
        if target is None:
            raise ValueError("out_dir not given (here or in BenchmarkSuite())")
        for b in self.runs:
            b.save(self._run_path(target, b))
        self._save_tables(target)
        return target
