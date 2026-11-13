"""Metody własne projektu. Klasyfikacja: HYPER z a priori zmieszanym z rozkładem
jednostajnym, opcjonalnie z regułą GLEE. Regresja: IB pruning z informatywnym a priori
wariancji wyjść drzew. Opis: implementacja.md, rozdz. 12."""
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestRegressor

from generate_table_IB import pi_tag

from .base import ORDER_SUFFIX, FloatArray, IntArray, Multiclass, Order
from .glee import tablica_glee
from .hyper import Hyper, prior_jednostajny
from .oob import wariancje_oob
from .regression import SequentialRegression, mnoznik_studenta


def dolacz_glee(prog: IntArray, prog_glee: IntArray, para: bool) -> IntArray:
    """Tablica reguły "HYPER lub GLEE": w każdej komórce mniejszy z dwóch progów głosów
    lidera. prog ma format tablica_hyper albo (para=True) tablica_par."""
    T = prog.shape[1] - 1
    if para:
        t2 = np.arange(T + 1)[None, :]
    else:                                           # druga klasa: najwięcej głosów poza liderem
        t2 = np.indices(prog.shape[1:]).max(axis=0)[None]
    # prog_glee ma T // 2 + 1 elementów: druga klasa nie ma więcej niż połowy głosów
    glee = np.where(t2 < len(prog_glee), prog_glee[np.minimum(t2, len(prog_glee) - 1)], T + 1)
    return np.minimum(prog, glee.astype(prog.dtype))


class Own(Hyper):
    """HYPER z a priori lam * P_OOB + (1 - lam) * P_jednostajny (lam = 1: HYPER, lam = 0:
    IB pruning); glee=True dodaje alternatywę z regułą GLEE."""

    PREFIX = "OWN"

    def __init__(self, pi: float = 0.99, lam: float = 0.5, glee: bool = False,
                 pi_glee: float | None = None, min_votes: int | None = None,
                 n_prior: int | None = 20000, smooth: int = 5, name: str | None = None,
                 order: Order = "fixed", seed: int | None = None,
                 multiclass: Multiclass = "full") -> None:
        if not 0 <= lam <= 1:
            raise ValueError(f"Own: lam musi być w [0, 1], a jest {lam}")
        pi_glee = pi if pi_glee is None else pi_glee
        if glee and not 0.5 < pi_glee <= 1:
            raise ValueError(f"Own: reguła GLEE wymaga poziomu ufności w (0.5, 1], a jest {pi_glee}")
        if min_votes is not None and min_votes < 1:
            raise ValueError(f"Own: min_votes musi być >= 1, a jest {min_votes}")
        self.lam, self.glee, self.pi_glee, self.min_votes = lam, glee, pi_glee, min_votes
        if name is None:
            name = (f"OWN_lam{lam:g}_pi{pi_tag(pi)}" + ("_glee" if glee else "")
                    + ("_pair" if multiclass == "pair" else "") + ORDER_SUFFIX[order])
        super().__init__(pi, n_prior, smooth, name, order, seed, multiclass)

    def _prior(self, prior: FloatArray) -> FloatArray:
        para = self.multiclass == "pair"
        return self.lam * prior + (1 - self.lam) * prior_jednostajny(self.T, self.l, para)

    def _prog(self, prog: IntArray) -> IntArray:
        if not self.glee:
            return prog
        alfa = round(1 - self.pi_glee, 12)          # bez błędu zaokrągleń 1 - 0.99
        return dolacz_glee(prog, tablica_glee(self.T, alfa, self.min_votes),
                           self.multiclass == "pair")


class OwnRegression(SequentialRegression):
    """Regresyjny IB pruning z a priori wariancji wyjść: nu0 pseudoobserwacji o wariancji
    sigma0_2, czyli medianie wariancji wyjść out-of-bag (nu0 = 0: zwykły IBRegression)."""

    PREFIX = "OWNREG"

    def __init__(self, pi: float = 0.99, delta: float | None = None,
                 delta_std: float | None = None, nu0: float = 5.0, t_min: int = 1,
                 n_prior: int | None = 20000, name: str | None = None,
                 order: Order = "fixed", seed: int | None = None) -> None:
        if nu0 < 0:
            raise ValueError(f"OwnRegression: nu0 musi być >= 0, a jest {nu0}")
        self.nu0, self.n_prior = nu0, n_prior
        if name is None:
            tol = f"d{delta:g}" if delta is not None else f"ds{delta_std:g}"
            name = f"OWNREG_nu{nu0:g}_pi{pi_tag(pi)}_{tol}{ORDER_SUFFIX[order]}"
        super().__init__(pi, delta, delta_std, t_min, name, order, seed)

    def fit(self, rf: RandomForestRegressor, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None) -> Self:
        if self.nu0 > 0 and isinstance(rf, RandomForestRegressor):  # inny las odrzuci super().fit
            if X_train is None:
                raise ValueError(f"{self.name}: fit(rf, X_train) wymaga danych uczących lasu "
                                 f"(wariancja wyjść out-of-bag)")
            wariancje = wariancje_oob(rf, X_train, self.n_prior)
            if len(wariancje) == 0:
                raise ValueError(f"{self.name}: żadna próbka ucząca nie ma dwóch drzew out-of-bag")
            self.sigma0_2 = float(np.median(wariancje))
        return super().fit(rf, X_train, y_train)

    def _mnoznik(self) -> FloatArray:
        return mnoznik_studenta(self.T, self.pi, self.t_min, self.nu0)
