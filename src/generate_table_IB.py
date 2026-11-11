"""
Generowanie tablicy stopu dla statystycznego instance-based pruningu
(Hernández-Lobato, Martínez-Muñoz, Suárez, IEEE TPAMI 31(2), 2009).

Tablica zależy tylko od T (liczba klasyfikatorów), l (liczba klas)
i pi (poziom ufności), więc wystarczy policzyć ją raz i zapisać do pliku.
"""
import sys
import time
from functools import lru_cache
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np


@lru_cache(maxsize=None)
def wygrana(t, T):
    """Prawdopodobieństwo, że klasa 0 ściśle wygra po zebraniu T głosów,
    jeśli dotychczasowe głosy to t (krotka liczb głosów na każdą klasę).

    Model urny Polyi: następny głos trafia do klasy i z prawdopodobieństwem
    (t[i] + 1) / (sum(t) + l). Dzięki temu wynik dla stanu t składa się
    z wyników dla stanów o jeden głos dalej, a lru_cache sprawia,
    że każdy stan liczony jest tylko raz.
    """
    if sum(t) == T:
        return float(t[0] > max(t[1:]))

    mianownik = sum(t) + len(t)
    wynik = 0.0
    for i in range(len(t)):
        nastepny = t[:i] + (t[i] + 1,) + t[i + 1:]
        wynik += (t[i] + 1) / mianownik * wygrana(nastepny, T)
    return wynik


def p_same(t, T):
    """Prawdopodobieństwo, że klasa prowadząca po zebraniu głosów t
    wygra też po zebraniu wszystkich T głosów (wzór (8) z artykułu).

    Sortujemy t malejąco, żeby klasa prowadząca była na pozycji 0
    (wynik nie zależy od kolejności klas, a cache jest lepiej wykorzystany).
    """
    # głębokość rekurencji ~T; bez tego dla dużego T zostanie przekroczony limit
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 3 * T + 100))
    return wygrana(tuple(sorted(map(int, t), reverse=True)), T)


def generuj_tablice(T, l, pi):
    """Generuje tablicę progów stopu dla zespołu T klasyfikatorów i l klas
    przy poziomie ufności pi i zapisuje ją do data/tablice/T{T}_L{l}_pi{pi*100}.npz.

    Przy ustalonych głosach pozostałych klas P(t, T) rośnie wraz z liczbą
    głosów klasy prowadzącej, więc zamiast pełnej tablicy (T+1)^l wartości
    True/False wystarczy zapamiętać próg:

        prog[t_2, ..., t_l] = najmniejsze t_1, dla którego P(t, T) >= pi
                              (T + 1, jeśli przy tych t_2..t_l nie da się przerwać)

    gdzie t_1 >= t_2 >= ... >= t_l to głosy posortowane malejąco.
    Wypełniane są tylko komórki o indeksach posortowanych malejąco,
    dlatego przed sprawdzeniem trzeba posortować głosy (patrz czy_stop).
    """
    start = time.perf_counter()
    prog = np.full((T + 1,) * (l - 1), T + 1, dtype=np.min_scalar_type(T + 1))
    for reszta in combinations_with_replacement(range(T + 1), l - 1):  # rosnąco
        reszta = reszta[::-1]                 # t_2 >= t_3 >= ... >= t_l
        for t1 in range(reszta[0], T - sum(reszta) + 1):
            if p_same((t1,) + reszta, T) >= pi:
                prog[reszta] = t1
                break
    wygrana.cache_clear()                     # zwolnienie pamięci po obliczeniach
    czas = time.perf_counter() - start        # czas generowania w sekundach

    out_dir = Path(__file__).resolve().parents[1] / "data" / "tablice"
    out_dir.mkdir(parents=True, exist_ok=True)
    plik = out_dir / f"T{T}_L{l}_pi{round(pi * 100, 6):g}.npz"
    np.savez_compressed(plik, prog=prog, T=T, l=l, pi=pi, time=czas)
    print(f"Saved tablice/{plik.name}: shape {prog.shape}, time {czas:.2f} s")
    return plik


def czy_stop(glosy, prog):
    """True, jeśli przy aktualnych głosach (liczba głosów na każdą klasę,
    w dowolnej kolejności) można przerwać odpytywanie klasyfikatorów."""
    t = sorted(map(int, glosy), reverse=True)
    return t[0] >= prog[tuple(t[1:])]


if __name__ == "__main__":
    generuj_tablice(T=51, l=2, pi=0.95)

    