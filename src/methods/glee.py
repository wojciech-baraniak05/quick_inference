"""GLEE: Gaussian Lazy Ensemble Evaluation, reguła wczesnego stopu z artykułu Basilico,
Munson, Kolda, Dixon, Kegelmeyer, "COMET: A Recipe for Learning and Using Large Ensembles
on Massive Data" (ICDM 2011, arXiv:1103.2068), rozdz. II-C. Opis: implementacja.md, rozdz. 6."""
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri
from sklearn.ensemble import RandomForestClassifier

from .base import (ORDER_SUFFIX, BoolArray, IntArray, Order, SequentialPruning, pi_tag,
                   stop_pair, stop_pair_one)


def alpha_from_pi(pi: float) -> float:
    """alpha = 1 - pi bez błędu zaokrągleń (1 - 0.99 daje 0.010000000000000009)."""
    return round(1 - pi, 12)


def default_min_votes(alpha: float) -> int:
    """Minimalna liczba głosów przed stopem zalecana w artykule: 15, 30 i 45 dla
    alpha = 1e-2, 1e-3 i 1e-4 (dla pośrednich alpha wartość ostrzejsza)."""
    if alpha >= 1e-2:
        return 15
    return 30 if alpha >= 1e-3 else 45


def glee_table(T: int, alpha: float, min_votes: int | None = None, fpc: bool = True,
                 one_sided: bool = True) -> IntArray:
    """threshold[t_2] = najmniejsze t_1, przy którym wolno przerwać (T + 1: nie da się);
    t_1 >= t_2 to głosy klasy prowadzącej i drugiej w kolejności."""
    if min_votes is None:
        min_votes = default_min_votes(alpha)
    z = ndtri(1 - (alpha if one_sided else alpha / 2))
    t2 = np.arange(T // 2 + 1)[:, None]
    t1 = np.arange(T + 1)[None, :]
    n = t1 + t2
    with np.errstate(divide="ignore", invalid="ignore"):
        p = t1 / n
        delta = z * np.sqrt(p * (1 - p) / n)
        if fpc:
            rho = np.sqrt(np.clip(T - n, 0, None) / max(T - 1, 1))
            delta = delta * np.where(n > 0.05 * T, rho, 1.0)
        stop = (n >= min_votes) & (n <= T) & (p - delta > 0.5)   # nan (alpha = 0) -> False
    threshold = np.where(stop.any(axis=1), stop.argmax(axis=1), T + 1)
    return threshold.astype(np.min_scalar_type(T + 1))


class Glee(SequentialPruning):
    """Stop, gdy dolna granica przedziału ufności dla odsetka głosów lidera przekroczy 0.5.
    Przedział opisuje losową próbę drzew, więc domyślnie każda próbka odpytuje drzewa w
    kolejności losowej bez zwracania (algorytm 2 artykułu); seed = 0 daje powtarzalne wyniki."""

    pi: float
    threshold: IntArray

    def __init__(self, pi: float = 0.99, min_votes: int | None = None, fpc: bool = True,
                 one_sided: bool = True, name: str | None = None,
                 order: Order = "random", seed: int | None = 0) -> None:
        if not 0.5 < pi <= 1:
            raise ValueError(f"{type(self).__name__}: pi must be in (0.5, 1], got {pi}")
        if min_votes is not None and min_votes < 1:
            raise ValueError(f"{type(self).__name__}: min_votes must be >= 1, got {min_votes}")
        self.pi, self.alpha = pi, alpha_from_pi(pi)
        self.min_votes, self.fpc, self.one_sided = min_votes, fpc, one_sided
        super().__init__(name or f"GLEE_pi{pi_tag(pi)}{ORDER_SUFFIX.get(order, '')}",
                         order=order, seed=seed)

    def fit(self, rf: RandomForestClassifier, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None, X_val: ArrayLike | None = None,
            y_val: ArrayLike | None = None) -> Self:
        super().fit(rf)
        self.threshold = glee_table(self.T, self.alpha, self.min_votes, self.fpc, self.one_sided)
        return self

    def _stop_batch(self, votes: IntArray) -> BoolArray:
        return stop_pair(votes, self.threshold)

    def _stop_one(self, votes: IntArray) -> bool:
        return stop_pair_one(votes, self.threshold)
