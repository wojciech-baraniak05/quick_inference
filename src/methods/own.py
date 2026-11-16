"""Metody własne projektu. Klasyfikacja: HYPER z a priori zmieszanym z rozkładem
jednostajnym, opcjonalnie z regułą GLEE. Regresja: IB pruning z informatywnym a priori
wariancji wyjść drzew. Opis: implementacja.md, rozdz. 12."""
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestRegressor

from .base import CALIB_SUFFIX, ORDER_SUFFIX, Calib, FloatArray, IntArray, Multiclass, Order, pi_tag
from .glee import alpha_from_pi, glee_table
from .hyper import SMOOTH_SUFFIX, Hyper, SmoothMode, uniform_prior
from .oob import oob_variances
from .regression import SequentialRegression, student_multiplier


def merge_glee(threshold: IntArray, glee_threshold: IntArray, pair: bool) -> IntArray:
    """Tablica reguły "HYPER lub GLEE": w każdej komórce mniejszy z dwóch progów głosów
    lidera. threshold ma format hyper_table albo (pair=True) pair_table."""
    T = threshold.shape[1] - 1
    if pair:
        t2 = np.arange(T + 1)[None, :]
    else:                                           # druga klasa: najwięcej głosów poza liderem
        t2 = np.indices(threshold.shape[1:]).max(axis=0)[None]
    # glee_threshold ma T // 2 + 1 elementów: druga klasa nie ma więcej niż połowy głosów
    n = len(glee_threshold)
    glee = np.where(t2 < n, glee_threshold[np.minimum(t2, n - 1)], T + 1)
    return np.minimum(threshold, glee.astype(threshold.dtype))


class Own(Hyper):
    """HYPER z a priori lam * P_OOB + (1 - lam) * P_jednostajny (lam = 1: HYPER, lam = 0:
    IB pruning); glee=True dodaje alternatywę z regułą GLEE. calib="val": P_OOB zastępuje
    a priori z głosów na zbiorze walidacyjnym. smooth_mode jak w Hyper."""

    PREFIX = "OWN"

    def __init__(self, pi: float = 0.99, lam: float = 0.5, glee: bool = False,
                 pi_glee: float | None = None, min_votes: int | None = None,
                 n_prior: int | None = 20000, smooth: int = 5, name: str | None = None,
                 order: Order = "fixed", seed: int | None = None,
                 multiclass: Multiclass = "full", calib: Calib = "oob",
                 smooth_mode: SmoothMode = "mean") -> None:
        if not 0 <= lam <= 1:
            raise ValueError(f"{type(self).__name__}: lam must be in [0, 1], got {lam}")
        pi_glee = pi if pi_glee is None else pi_glee
        if glee and not 0.5 < pi_glee <= 1:
            raise ValueError(f"{type(self).__name__}: GLEE rule needs pi_glee in (0.5, 1], "
                             f"got {pi_glee}")
        if min_votes is not None and min_votes < 1:
            raise ValueError(f"{type(self).__name__}: min_votes must be >= 1, got {min_votes}")
        self.lam, self.glee, self.pi_glee, self.min_votes = lam, glee, pi_glee, min_votes
        if name is None:
            name = (f"OWN_lam{lam:g}_pi{pi_tag(pi)}" + ("_glee" if glee else "")
                    + ("_pair" if multiclass == "pair" else "")
                    + SMOOTH_SUFFIX.get(smooth_mode, "") + CALIB_SUFFIX.get(calib, "")
                    + ORDER_SUFFIX.get(order, ""))
        super().__init__(pi, n_prior, smooth, name, order, seed, multiclass, calib, smooth_mode)

    def _prior(self, prior: FloatArray) -> FloatArray:
        pair = self.multiclass == "pair"
        return self.lam * prior + (1 - self.lam) * uniform_prior(self.T, self.l, pair)

    def _threshold(self, threshold: IntArray) -> IntArray:
        if not self.glee:
            return threshold
        glee = glee_table(self.T, alpha_from_pi(self.pi_glee), self.min_votes)
        return merge_glee(threshold, glee, self.multiclass == "pair")


class OwnRegression(SequentialRegression):
    """Regresyjny IB pruning z a priori wariancji wyjść: nu0 pseudoobserwacji o wariancji
    sigma0_2, czyli medianie wariancji wyjść out-of-bag (nu0 = 0: zwykły IBRegression)."""

    PREFIX = "OWNREG"

    def __init__(self, pi: float = 0.99, delta: float | None = None,
                 delta_std: float | None = None, nu0: float = 5.0, t_min: int = 1,
                 n_prior: int | None = 20000, name: str | None = None,
                 order: Order = "fixed", seed: int | None = None) -> None:
        if nu0 < 0:
            raise ValueError(f"{type(self).__name__}: nu0 must be >= 0, got {nu0}")
        self.nu0, self.n_prior = nu0, n_prior
        if name is None:
            tol = f"d{delta:g}" if delta is not None else f"ds{delta_std:g}"
            name = f"OWNREG_nu{nu0:g}_pi{pi_tag(pi)}_{tol}{ORDER_SUFFIX.get(order, '')}"
        super().__init__(pi, delta, delta_std, t_min, name, order, seed)

    def fit(self, rf: RandomForestRegressor, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None, X_val: ArrayLike | None = None,
            y_val: ArrayLike | None = None) -> Self:
        if self.nu0 > 0 and isinstance(rf, RandomForestRegressor):  # inny las odrzuci super().fit
            if X_train is None:
                raise ValueError(f"{self.name}: fit(rf, X_train) needs the forest's training data "
                                 f"(out-of-bag output variance)")
            variances = oob_variances(rf, X_train, self.n_prior)
            if len(variances) == 0:
                raise ValueError(f"{self.name}: no training sample has two out-of-bag trees")
            self.sigma0_2 = float(np.median(variances))
        return super().fit(rf, X_train, y_train)

    def _multiplier(self) -> FloatArray:
        return student_multiplier(self.T, self.pi, self.t_min, self.nu0)
