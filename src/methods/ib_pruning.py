"""Statystyczny instance-based pruning (Hernández-Lobato, Martínez-Muñoz, Suárez,
IEEE TPAMI 31(2), 2009), razem z generowaniem tablic progów stopu.
Model i format tablicy: implementacja.md, rozdz. 4."""
import sys
import time
from collections.abc import Iterable
from functools import lru_cache
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier

from .base import (MULTICLASS, ORDER_SUFFIX, BoolArray, IntArray, Multiclass, Order,
                   SequentialPruning, pi_tag, stop_pair, stop_pair_one)

TABLE_DIR = Path(__file__).resolve().parents[2] / "data" / "tablice"

Votes = tuple[int, ...]         # liczba głosów na każdą klasę


def table_path(T: int, l: int, pi: float) -> Path:
    return TABLE_DIR / f"T{T}_L{l}_pi{pi_tag(pi)}.npz"


@lru_cache(maxsize=None)
def win_probability(t: Votes, T: int) -> float:
    """P(klasa 0 ściśle wygra po T głosach | dotychczasowe głosy t) w modelu urny Polyi."""
    if sum(t) == T:
        return float(t[0] > max(t[1:]))

    denominator = sum(t) + len(t)
    result = 0.0
    for i in range(len(t)):
        next_votes = t[:i] + (t[i] + 1,) + t[i + 1:]
        result += (t[i] + 1) / denominator * win_probability(next_votes, T)
    return result


def p_same(t: Iterable[int], T: int) -> float:
    """P(klasa prowadząca w t wygra też po wszystkich T głosach), wzór (8) z artykułu."""
    # głębokość rekurencji to ok. T, domyślny limit nie wystarcza dla dużych lasów
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 3 * T + 100))
    # klasa prowadząca na pozycji 0; sortowanie zmniejsza też liczbę stanów w cache
    return win_probability(tuple(sorted(map(int, t), reverse=True)), T)


def generate_table(T: int, l: int, pi: float) -> Path:
    """Liczy tablicę progów i zapisuje ją do data/tablice/T{T}_L{l}_pi{pi*100}.npz."""
    start = time.perf_counter()
    threshold = np.full((T + 1,) * (l - 1), T + 1, dtype=np.min_scalar_type(T + 1))
    for rest in combinations_with_replacement(range(T + 1), l - 1):    # rosnąco
        rest = rest[::-1]                     # t_2 >= t_3 >= ... >= t_l
        for t1 in range(rest[0], T - sum(rest) + 1):
            if p_same((t1,) + rest, T) >= pi:
                threshold[rest] = t1
                break
    win_probability.cache_clear()             # zwolnienie pamięci po obliczeniach
    elapsed = time.perf_counter() - start

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = table_path(T, l, pi)
    # klucz "prog" w pliku zostaje dla zgodności z tablicami w repozytorium
    np.savez_compressed(path, prog=threshold, T=T, l=l, pi=pi, time=elapsed)
    print(f"saved tablice/{path.name}: shape {threshold.shape}, {elapsed:.2f} s")
    return path


def stop_ib_one(votes: Iterable[int], threshold: NDArray[np.integer[Any]]) -> bool:
    """Czy przy tych głosach (klasy w dowolnej kolejności) można przerwać głosowanie."""
    t = sorted(map(int, votes), reverse=True)
    return bool(t[0] >= threshold[tuple(t[1:])])


@lru_cache(maxsize=None)
def _load_npz(path: str) -> tuple[IntArray, float]:
    d = np.load(path)
    return d["prog"], float(d["pi"])


def load_table(table: str | Path, hint: str = "") -> tuple[IntArray, float]:
    """(threshold, pi) z pliku .npz; ścieżka względna jest liczona od data/tablice/."""
    path = TABLE_DIR / table                        # ścieżka bezwzględna zostaje bez zmian
    if not path.exists():
        root = TABLE_DIR.parents[1]                 # w komunikacie ścieżka względem repozytorium
        shown = path.relative_to(root).as_posix() if path.is_relative_to(root) else path
        raise FileNotFoundError(f"threshold table not found: {shown}{hint}")
    return _load_npz(str(path))


def stop_ib(votes: IntArray, threshold: IntArray) -> BoolArray:
    t = -np.sort(-votes, axis=1)                    # głosy malejąco w każdym wierszu
    return t[:, 0] >= threshold[tuple(t[:, 1:].T)]


class IBPruning(SequentialPruning):
    """Stop, gdy klasa prowadząca wygra pełne głosowanie z prawdopodobieństwem >= pi.
    multiclass="pair": przy wielu klasach tablica binarna dla lidera i drugiej klasy."""

    threshold: IntArray

    def __init__(self, threshold: ArrayLike | None = None, pi: float | None = None,
                 table: str | Path | None = None, name: str | None = None,
                 order: Order = "fixed", seed: int | None = None,
                 multiclass: Multiclass = "full") -> None:
        if multiclass not in MULTICLASS:
            raise ValueError(f"{type(self).__name__}: multiclass must be one of {MULTICLASS}, "
                             f"got {multiclass!r}")
        self.multiclass = multiclass
        if table is not None:
            threshold, pi = load_table(table)
        if threshold is None and pi is None:
            raise ValueError(f"{type(self).__name__}: give pi, table or threshold")
        self._threshold = None if threshold is None else np.asarray(threshold)
        self.pi = pi
        if name is None:
            name = f"IB_pi{pi_tag(pi)}" if pi is not None else "ib_pruning"
            name += ("_pair" if multiclass == "pair" else "") + ORDER_SUFFIX.get(order, "")
        super().__init__(name, order=order, seed=seed)

    def fit(self, rf: RandomForestClassifier, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None, X_val: ArrayLike | None = None,
            y_val: ArrayLike | None = None) -> Self:
        super().fit(rf)
        pair = self.multiclass == "pair"
        l = 2 if pair else self.l                   # liczba klas, dla której jest tablica
        if self._threshold is not None:
            self.threshold = self._threshold
        else:                                       # tablica dobierana do rozmiaru lasu
            assert self.pi is not None
            hint = "" if l == 2 else ' or set multiclass = "pair" (binary table)'
            self.threshold = load_table(
                table_path(self.T, l, self.pi),
                f"; create it with generate_table(T={self.T}, l={l}, pi={self.pi}) "
                f"from methods.ib_pruning{hint}")[0]
        T_table, l_table = self.threshold.shape[0] - 1, self.threshold.ndim + 1
        if (T_table, l_table) != (self.T, l):
            raise ValueError(f"{self.name}: table is for T={T_table}, l={l_table}, "
                             f"forest has T={self.T}, l={self.l}"
                             + (' (multiclass="pair" needs an l=2 table)' if pair else ""))
        return self

    def _stop_batch(self, votes: IntArray) -> BoolArray:
        if self.multiclass == "pair":
            return stop_pair(votes, self.threshold)
        return stop_ib(votes, self.threshold)

    def _stop_one(self, votes: IntArray) -> bool:
        if self.multiclass == "pair":
            return stop_pair_one(votes, self.threshold)
        return stop_ib_one(votes, self.threshold)
