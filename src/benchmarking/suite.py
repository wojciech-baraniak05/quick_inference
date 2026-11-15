"""Zapis wielu benchmarków (zbiory danych, powtórzenia) do pliku wyników
(opis: implementacja.md, rozdz. 9.4)."""
from pathlib import Path
from typing import Self

import pandas as pd

from .benchmark import Benchmark
from .storage import store_runs


class BenchmarkSuite:
    """Dopisuje każdy dodany benchmark do pliku wyników od razu po policzeniu. Wyników nie
    trzyma: są w pliku, a metryki liczy src/analysis."""

    def __init__(self, out_file: str | Path) -> None:
        self.out_file = Path(out_file)
        self.n_runs = 0
        self._unsaved: list[pd.DataFrame] = []      # przebiegi, których zapis się nie udał

    def add(self, b: Benchmark) -> Self:
        """Dopisuje przebieg do pliku. Błąd zapisu (np. plik zablokowany przez inny program)
        nie przerywa benchmarku: przebieg czeka i jest dopisywany przy następnym add()
        albo w save()."""
        self._unsaved.append(b.to_frame())
        self.n_runs += 1
        try:
            self.save()
        except OSError as e:
            print(f"  results not written ({e}); retried on next add()/save()")
        return self

    def save(self) -> Path:
        """Dopisuje zaległe przebiegi (te, których zapis w add() się nie udał)."""
        if self._unsaved:
            store_runs(self.out_file, self._unsaved)
            self._unsaved.clear()
        return self.out_file

    def __len__(self) -> int:
        return self.n_runs
