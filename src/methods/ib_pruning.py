"""Statystyczny instance-based pruning (Hernández-Lobato i in., 2009).
Opis: implementacja.md, rozdz. 4."""
from functools import lru_cache
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier

from generate_table_IB import TABLE_DIR, czy_stop, pi_tag, plik_tablicy

from .base import BoolArray, IntArray, Order, SequentialPruning


@lru_cache(maxsize=None)
def _wczytaj_npz(path: str) -> tuple[IntArray, float]:
    d = np.load(path)
    return d["prog"], float(d["pi"])


def wczytaj_tablice(plik: str | Path, podpowiedz: str = "") -> tuple[IntArray, float]:
    """(prog, pi) z pliku .npz; ścieżka względna jest liczona od data/tablice/."""
    path = TABLE_DIR / plik                         # ścieżka bezwzględna zostaje bez zmian
    if not path.exists():
        raise FileNotFoundError(f"brak tablicy progów {path}{podpowiedz}")
    return _wczytaj_npz(str(path))


def czy_stop_wektorowo(votes: IntArray, prog: IntArray) -> BoolArray:
    t = -np.sort(-votes, axis=1)                    # głosy malejąco w każdym wierszu
    return t[:, 0] >= prog[tuple(t[:, 1:].T)]


class IBPruning(SequentialPruning):
    """Stop, gdy klasa prowadząca wygra pełne głosowanie z prawdopodobieństwem >= pi."""

    prog: IntArray

    def __init__(self, prog: ArrayLike | None = None, pi: float | None = None,
                 table: str | Path | None = None, name: str | None = None,
                 order: Order = "fixed", seed: int | None = None) -> None:
        if table is not None:
            prog, pi = wczytaj_tablice(table)
        if prog is None and pi is None:
            raise ValueError("IBPruning: podaj pi, table albo prog")
        self._prog = None if prog is None else np.asarray(prog)
        self.pi = pi
        if name is None:
            name = f"IB_pi{pi_tag(pi)}" if pi is not None else "ib_pruning"
            if order == "random":
                name += "_rand"
        super().__init__(name, order=order, seed=seed)

    def fit(self, rf: RandomForestClassifier, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None) -> Self:
        super().fit(rf)
        if self._prog is not None:
            self.prog = self._prog
        else:                                       # tablica dobierana do rozmiaru lasu
            assert self.pi is not None
            self.prog = wczytaj_tablice(
                plik_tablicy(self.T, self.l, self.pi),
                f"; wygeneruj ją w src/: from generate_table_IB import generuj_tablice; "
                f"generuj_tablice(T={self.T}, l={self.l}, pi={self.pi})")[0]
        T_prog, l_prog = self.prog.shape[0] - 1, self.prog.ndim + 1
        if (T_prog, l_prog) != (self.T, self.l):
            raise ValueError(f"{self.name}: tablica jest dla T={T_prog}, l={l_prog}, "
                             f"a las ma T={self.T}, l={self.l}")
        return self

    def _stop_batch(self, votes: IntArray) -> BoolArray:
        return czy_stop_wektorowo(votes, self.prog)

    def _stop_one(self, votes: IntArray) -> bool:
        return czy_stop(votes, self.prog)
