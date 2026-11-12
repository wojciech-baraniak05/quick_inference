"""Tablice progów stopu dla statystycznego instance-based pruningu
(Hernández-Lobato, Martínez-Muñoz, Suárez, IEEE TPAMI 31(2), 2009).
Model i format tablicy: implementacja.md, rozdz. 4."""
import sys
import time
from collections.abc import Iterable
from functools import lru_cache
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

TABLE_DIR = Path(__file__).resolve().parents[1] / "data" / "tablice"

Votes = tuple[int, ...]         # liczba głosów na każdą klasę


def pi_tag(pi: float) -> str:
    """Poziom ufności w procentach do nazw plików i metod: 0.98 -> '98', 0.995 -> '99.5'."""
    return f"{round(pi * 100, 6):g}"


def plik_tablicy(T: int, l: int, pi: float) -> Path:
    return TABLE_DIR / f"T{T}_L{l}_pi{pi_tag(pi)}.npz"


@lru_cache(maxsize=None)
def wygrana(t: Votes, T: int) -> float:
    """P(klasa 0 ściśle wygra po T głosach | dotychczasowe głosy t) w modelu urny Polyi."""
    if sum(t) == T:
        return float(t[0] > max(t[1:]))

    mianownik = sum(t) + len(t)
    wynik = 0.0
    for i in range(len(t)):
        nastepny = t[:i] + (t[i] + 1,) + t[i + 1:]
        wynik += (t[i] + 1) / mianownik * wygrana(nastepny, T)
    return wynik


def p_same(t: Iterable[int], T: int) -> float:
    """P(klasa prowadząca w t wygra też po wszystkich T głosach), wzór (8) z artykułu."""
    # głębokość rekurencji to ok. T, domyślny limit nie wystarcza dla dużych lasów
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 3 * T + 100))
    # klasa prowadząca na pozycji 0; sortowanie zmniejsza też liczbę stanów w cache
    return wygrana(tuple(sorted(map(int, t), reverse=True)), T)


def generuj_tablice(T: int, l: int, pi: float) -> Path:
    """Liczy tablicę progów i zapisuje ją do data/tablice/T{T}_L{l}_pi{pi*100}.npz."""
    start = time.perf_counter()
    prog = np.full((T + 1,) * (l - 1), T + 1, dtype=np.min_scalar_type(T + 1))
    for reszta in combinations_with_replacement(range(T + 1), l - 1):  # rosnąco
        reszta = reszta[::-1]                 # t_2 >= t_3 >= ... >= t_l
        for t1 in range(reszta[0], T - sum(reszta) + 1):
            if p_same((t1,) + reszta, T) >= pi:
                prog[reszta] = t1
                break
    wygrana.cache_clear()                     # zwolnienie pamięci po obliczeniach
    czas = time.perf_counter() - start

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plik = plik_tablicy(T, l, pi)
    np.savez_compressed(plik, prog=prog, T=T, l=l, pi=pi, time=czas)
    print(f"Saved tablice/{plik.name}: shape {prog.shape}, time {czas:.2f} s")
    return plik


def czy_stop(glosy: Iterable[int], prog: NDArray[np.integer[Any]]) -> bool:
    """Czy przy tych głosach (klasy w dowolnej kolejności) można przerwać głosowanie."""
    t = sorted(map(int, glosy), reverse=True)
    return bool(t[0] >= prog[tuple(t[1:])])


if __name__ == "__main__":
    generuj_tablice(T=51, l=2, pi=0.95)
