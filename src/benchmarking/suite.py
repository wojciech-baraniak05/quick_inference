"""Wyniki wielu benchmarków: wiele zbiorów danych i powtórzeń (opis: implementacja.md, rozdz. 9)."""
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Self

import pandas as pd

from .benchmark import Benchmark
from .metrics import ARRAY_COLS, ROW_COLS
from .modes import SUPPORTED_MODES, Mode


def _run_ids(df: pd.DataFrame, cols: list[str]) -> list[tuple[object, ...]]:
    """Tożsamość przebiegu w każdym wierszu: krotka wartości cols, brak wartości jako None
    (np. kolumna, której starsze przebiegi nie miały)."""
    vals = df.reindex(columns=cols).astype(object)
    vals = vals.where(vals.notna(), None)
    return list(map(tuple, vals.to_numpy()))


def store_runs(path: str | Path, frames: Iterable[pd.DataFrame]) -> Path:
    """Dopisuje przebiegi (Benchmark.to_frame()) do pliku Parquet. Wiersz zastępuje zapisany
    wiersz o tych samych ustawieniach przebiegu (dataset, seed, n_trees...; bez date),
    scenariuszu i metodzie. Pozostałe wiersze zostają, także metody, których nowy przebieg
    nie liczył. Zapis przez plik tymczasowy i podmianę."""
    path = Path(path)
    table = pd.read_parquet(path) if path.exists() else None
    for new in frames:                              # po kolei: późniejszy wygrywa
        if table is not None:
            run_cols = sorted({*table.columns, *new.columns} - {*ROW_COLS, *ARRAY_COLS})
            cols = [*run_cols, "mode", "method"]
            replaced = set(_run_ids(new, cols))
            table = table[[run not in replaced for run in _run_ids(table, cols)]]
        table = new if table is None else pd.concat([table, new], ignore_index=True)
    if table is None:
        raise ValueError("store_runs: no runs to store")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    table.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    return path


class BenchmarkSuite:
    """Benchmarki identyfikowane kluczami z metadanych; ostatni klucz to powtórzenie,
    po którym uśrednia aggregate()."""

    # kolumny jakości zależą od zadania; summary() pokazuje te, które są w wynikach
    SUMMARY_COLS = ["accuracy", "f1", "mae", "rmse", "r2", "agree_rf", "agree_full", "mad_full",
                    "ms_per_sample", "speedup_vs_rf", "speedup_vs_full", "trees_mean", "trees_frac"]

    def __init__(self, keys: Sequence[str] = ("dataset", "seed"),
                 out_file: str | Path | None = None) -> None:
        self.keys = list(keys)
        self.out_file = None if out_file is None else Path(out_file)
        self.runs: list[Benchmark] = []
        self._unsaved: list[Benchmark] = []         # przebiegi, których zapis się nie udał

    def add(self, b: Benchmark) -> Self:
        """Dodaje benchmark; z out_file od razu dopisuje go do pliku wyników."""
        missing = [k for k in self.keys if k not in b.metadata]
        if missing:
            raise ValueError(f"benchmark metadata is missing keys: {missing}")
        self.runs.append(b)
        if self.out_file is not None:
            self._unsaved.append(b)
            try:
                store_runs(self.out_file, [u.to_frame() for u in self._unsaved])
                self._unsaved.clear()
            except OSError as e:
                # np. plik zablokowany przez inny program; wyniki są w pamięci i zostaną
                # dopisane przy następnym add() albo w save()
                print(f"  results not written ({e}); retried on next add()/save()")
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

    def save(self, out_file: str | Path | None = None) -> Path:
        """Dopisuje wszystkie przebiegi do pliku wyników (domyślnie out_file z konstruktora);
        zapisane już przebiegi są zastępowane, więc wynik jest ten sam."""
        target = self.out_file if out_file is None else Path(out_file)
        if target is None:
            raise ValueError("out_file not given (here or in BenchmarkSuite())")
        store_runs(target, [b.to_frame() for b in self.runs])
        if target == self.out_file:
            self._unsaved.clear()
        return target
