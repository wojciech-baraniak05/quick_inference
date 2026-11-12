"""GLEE: Gaussian Lazy Ensemble Evaluation, reguła wczesnego stopu z artykułu Basilico,
Munson, Kolda, Dixon, Kegelmeyer, "COMET: A Recipe for Learning and Using Large Ensembles
on Massive Data" (ICDM 2011, arXiv:1103.2068), rozdz. II-C. Opis: implementacja.md, rozdz. 6."""
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri
from sklearn.ensemble import RandomForestClassifier

from generate_table_IB import pi_tag

from .base import BoolArray, IntArray, SequentialPruning


def min_glosow(alfa: float) -> int:
    """Minimalna liczba głosów przed stopem zalecana w artykule: 15, 30 i 45 dla
    alfa = 1e-2, 1e-3 i 1e-4 (dla pośrednich alfa wartość ostrzejsza)."""
    if alfa >= 1e-2:
        return 15
    return 30 if alfa >= 1e-3 else 45


def tablica_glee(T: int, alfa: float, min_votes: int | None = None, fpc: bool = True,
                 one_sided: bool = True) -> IntArray:
    """prog[t_2] = najmniejsze t_1, przy którym wolno przerwać (T + 1: nie da się);
    t_1 >= t_2 to głosy klasy prowadzącej i drugiej w kolejności."""
    if min_votes is None:
        min_votes = min_glosow(alfa)
    z = ndtri(1 - (alfa if one_sided else alfa / 2))
    t2 = np.arange(T // 2 + 1)[:, None]
    t1 = np.arange(T + 1)[None, :]
    n = t1 + t2
    with np.errstate(divide="ignore", invalid="ignore"):
        p = t1 / n
        delta = z * np.sqrt(p * (1 - p) / n)
        if fpc:
            rho = np.sqrt(np.clip(T - n, 0, None) / max(T - 1, 1))
            delta = delta * np.where(n > 0.05 * T, rho, 1.0)
        stop = (n >= min_votes) & (n <= T) & (p - delta > 0.5)   # nan (alfa = 0) -> False
    prog = np.where(stop.any(axis=1), stop.argmax(axis=1), T + 1)
    return prog.astype(np.min_scalar_type(T + 1))


def czy_stop_glee(votes: IntArray, prog: IntArray) -> BoolArray:
    t = np.partition(votes, -2, axis=1)             # dwie ostatnie kolumny: t_2, t_1
    return t[:, -1] >= prog[t[:, -2]]


class Glee(SequentialPruning):
    """Stop, gdy dolna granica przedziału ufności dla odsetka głosów lidera przekroczy 0.5."""

    pi: float
    prog: IntArray

    def __init__(self, pi: float = 0.99, min_votes: int | None = None, fpc: bool = True,
                 one_sided: bool = True, name: str | None = None) -> None:
        if not 0.5 < pi <= 1:
            raise ValueError(f"Glee: pi musi być w (0.5, 1], a jest {pi}")
        if min_votes is not None and min_votes < 1:
            raise ValueError(f"Glee: min_votes musi być >= 1, a jest {min_votes}")
        self.pi, self.alfa = pi, round(1 - pi, 12)      # bez błędu zaokrągleń 1 - 0.99
        self.min_votes, self.fpc, self.one_sided = min_votes, fpc, one_sided
        super().__init__(name or f"GLEE_pi{pi_tag(pi)}")

    def fit(self, rf: RandomForestClassifier, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None) -> Self:
        super().fit(rf)
        self.prog = tablica_glee(self.T, self.alfa, self.min_votes, self.fpc, self.one_sided)
        return self

    def _stop_batch(self, votes: IntArray) -> BoolArray:
        return czy_stop_glee(votes, self.prog)

    def _stop_one(self, votes: IntArray) -> bool:
        t = np.sort(votes)
        return bool(t[-1] >= self.prog[t[-2]])
