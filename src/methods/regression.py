"""Wczesny stop w regresji: metody odniesienia, wspólna pętla oraz regresyjne warianty
IB pruning i GLEE. Opis: implementacja.md, rozdz. 11."""
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri, stdtrit
from sklearn.ensemble import RandomForestRegressor

from generate_table_IB import pi_tag

from .base import (ORDER_SUFFIX, Features, FastRegTree, FloatArray, InferenceMethod, Order,
                   Prediction, SequentialPruning, Task, fast_reg_trees)
from .glee import min_glosow


class RegressionMethod(InferenceMethod):
    """Bazowa klasa metod dla lasu regresyjnego: predykcją jest średnia wyjść drzew."""

    task: Task = "regression"
    rf: RandomForestRegressor
    trees: list[FastRegTree]

    def fit(self, rf: RandomForestRegressor, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None) -> Self:
        if not isinstance(rf, RandomForestRegressor):
            raise ValueError(f"{self.name}: metoda regresyjna, a las to {type(rf).__name__}")
        self.rf = rf
        self.trees = fast_reg_trees(rf)
        self.classes = np.empty(0)
        self.T = len(self.trees)
        self.l = 0
        self.n_features = rf.n_features_in_
        return self


class SklearnRFReg(RegressionMethod):
    """Oryginalny rf.predict lasu regresyjnego."""

    name = "rf_sklearn"

    def predict_batch(self, X: ArrayLike) -> Prediction:
        y = self.rf.predict(X)
        return y, np.full(len(y), self.T)

    def predict_one(self, x: ArrayLike) -> tuple[Any, int]:
        return self.rf.predict(x)[0], self.T


class FullAveraging(RegressionMethod):
    """Średnia wyjść wszystkich drzew, liczona drzewo po drzewie."""

    name = "full_averaging"

    def predict_batch(self, X: ArrayLike) -> Prediction:
        X = self._X32(X)
        suma = np.zeros(len(X))
        for tree in self.trees:
            suma += tree.predict(X)
        return suma / self.T, np.full(len(X), self.T)

    def predict_one(self, x: ArrayLike) -> tuple[Any, int]:
        x = self._X32(x)
        suma = 0.0
        for tree in self.trees:
            suma += tree.predict(x)[0]
        return suma / self.T, self.T


def poprawka_fpc(T: int) -> FloatArray:
    """rho[t] = sqrt((T - t) / (T - 1)) dla t > 0.05 * T, inaczej 1 (t = 0..T)."""
    t = np.arange(T + 1)
    return np.where(t > 0.05 * T, np.sqrt((T - t) / max(T - 1, 1)), 1.0)


def mnoznik_studenta(T: int, pi: float, t_min: int = 2, nu0: float = 0.0) -> FloatArray:
    """c[t] = q_{nu0 + t - 1, (1 + pi) / 2} * sqrt((T - t) / (T * t)) dla t = 0..T; inf tam,
    gdzie po t zapytaniach nie wolno przerwać (t < t_min albo brak stopni swobody)."""
    t = np.arange(T + 1)
    df = nu0 + t - 1
    c = np.full(T + 1, np.inf)
    ok = (t >= max(t_min, 1)) & (df > 0)
    with np.errstate(invalid="ignore"):             # pi = 1: inf * 0 dla t = T
        c[ok] = stdtrit(df[ok], (1 + pi) / 2) * np.sqrt((T - t[ok]) / (T * t[ok]))
    return c


def mnoznik_normalny(T: int, pi: float, t_min: int = 2, fpc: bool = True) -> FloatArray:
    """c[t] = z_{(1 + pi) / 2} * rho[t] / sqrt(t) dla t = 0..T; inf dla t < t_min."""
    t = np.arange(T + 1)
    c = np.full(T + 1, np.inf)
    ok = t >= max(t_min, 2)                         # s_t wymaga co najmniej dwóch wyjść
    rho = poprawka_fpc(T) if fpc else np.ones(T + 1)
    with np.errstate(invalid="ignore"):             # pi = 1: inf * 0 dla t = T
        c[ok] = ndtri((1 + pi) / 2) * rho[ok] / np.sqrt(t[ok])
    return c


def prog_sumy_kwadratow(c: FloatArray, delta: float, nu0: float = 0.0,
                        sigma0_2: float = 0.0) -> FloatArray:
    """Warunek c[t] * sigma_t <= delta zapisany dla sumy kwadratów odchyleń S_t wyjść:
    stop, gdy S_t <= prog[t], gdzie sigma_t^2 = (nu0 * sigma0_2 + S_t) / (nu0 + t - 1).
    prog[t] = -1: po t zapytaniach nie wolno przerwać; prog[T] = inf."""
    T = len(c) - 1
    t = np.arange(T + 1)
    prog = np.full(T + 1, -1.0)
    ok = np.isfinite(c) & (c > 0)
    prog[ok] = (delta / c[ok]) ** 2 * (nu0 + t[ok] - 1) - nu0 * sigma0_2
    prog[c == 0] = np.inf
    prog[T] = np.inf                                # po wszystkich drzewach zawsze koniec
    return prog


class SequentialRegression(RegressionMethod):
    """Wczesny stop w regresji: drzewa odpowiadają po kolei, a próbka kończy, gdy
    mnoznik[t] * rozrzut wyjść <= delta. Podklasa podaje _mnoznik()."""

    ORDERS = SequentialPruning.ORDERS
    PREFIX = "REG"
    pi: float
    delta: float
    nu0: float = 0.0            # a priori wariancji wyjść: liczba pseudoobserwacji
    sigma0_2: float = 0.0       # i ich wariancja
    mnoznik: FloatArray         # c[t], t = 0..T
    prog_S: FloatArray          # próg sumy kwadratów odchyleń, t = 0..T
    rng: np.random.Generator

    def __init__(self, pi: float = 0.99, delta: float | None = None,
                 delta_std: float | None = None, t_min: int = 2, name: str | None = None,
                 order: Order = "fixed", seed: int | None = None) -> None:
        klasa = type(self).__name__
        if not 0 < pi <= 1:
            raise ValueError(f"{klasa}: pi musi być w (0, 1], a jest {pi}")
        if (delta is None) == (delta_std is None):
            raise ValueError(f"{klasa}: podaj dokładnie jedno z delta (tolerancja w jednostkach "
                             f"zmiennej celu) i delta_std (w odchyleniach standardowych y_train)")
        tolerancja = delta if delta is not None else delta_std
        assert tolerancja is not None
        if tolerancja <= 0:
            raise ValueError(f"{klasa}: tolerancja musi być dodatnia, a jest {tolerancja}")
        if t_min < 1:
            raise ValueError(f"{klasa}: t_min musi być >= 1, a jest {t_min}")
        if order not in self.ORDERS:
            raise ValueError(f"order musi być jednym z {self.ORDERS}, a jest {order!r}")
        self.pi, self._delta, self.delta_std, self.t_min = pi, delta, delta_std, t_min
        self.order, self.seed = order, seed
        if name is None:
            tol = f"d{delta:g}" if delta is not None else f"ds{delta_std:g}"
            name = f"{self.PREFIX}_pi{pi_tag(pi)}_{tol}{ORDER_SUFFIX[order]}"
        super().__init__(name)

    def _mnoznik(self) -> FloatArray:
        """Tablica c[t] dla t = 0..T; inf tam, gdzie po t zapytaniach nie wolno przerwać."""
        raise NotImplementedError

    def fit(self, rf: RandomForestRegressor, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None) -> Self:
        super().fit(rf)
        if self._delta is not None:
            self.delta = self._delta
        else:
            if y_train is None:
                raise ValueError(f"{self.name}: delta_std wymaga y_train w fit(rf, X_train, y_train)")
            assert self.delta_std is not None
            self.delta = self.delta_std * float(np.std(np.asarray(y_train, dtype=float)))
            if self.delta <= 0:
                raise ValueError(f"{self.name}: y_train ma zerowe odchylenie standardowe, "
                                 f"podaj delta zamiast delta_std")
        self.rng = np.random.default_rng(self.seed)
        if self.order == "shuffled":                # jedna permutacja lasu dla wszystkich próbek
            self.trees = [self.trees[j] for j in self.rng.permutation(self.T)]
        self.mnoznik = self._mnoznik()
        self.prog_S = prog_sumy_kwadratow(self.mnoznik, self.delta, self.nu0, self.sigma0_2)
        self._prog_lista: list[float] = self.prog_S.tolist()
        return self

    def predict_batch(self, X: ArrayLike) -> Prediction:
        X = self._X32(X)
        if self.order == "random":
            return self._predict_batch_random(X)
        y_pred = np.empty(len(X))
        n_used = np.full(len(X), self.T)
        aktywne = np.arange(len(X))                 # próbki jeszcze niezdecydowane
        X_akt = X                                   # ich cechy, średnie i sumy kwadratów
        srednia = np.zeros(len(X))
        S = np.zeros(len(X))
        for k, tree in enumerate(self.trees):
            y = tree.predict(X_akt)
            d = y - srednia
            srednia += d / (k + 1)
            S += d * (y - srednia)
            stop = S <= self.prog_S[k + 1]
            if stop.any():
                y_pred[aktywne[stop]] = srednia[stop]
                n_used[aktywne[stop]] = k + 1
                aktywne = aktywne[~stop]
                if len(aktywne) == 0:
                    break
                X_akt, srednia, S = X_akt[~stop], srednia[~stop], S[~stop]
        return y_pred, n_used

    def _predict_batch_random(self, X: Features) -> Prediction:
        n = len(X)
        n_used = np.full(n, self.T)
        srednia = np.zeros(n)
        S = np.zeros(n)
        y = np.empty(n)
        # perm[i, k] = drzewo, które próbka i pyta w kroku k
        perm = np.tile(np.arange(self.T, dtype=np.min_scalar_type(self.T)), (n, 1))
        perm = self.rng.permuted(perm, axis=1)
        aktywne = np.arange(n)
        for k in range(self.T):
            # próbki pytające w tym kroku to samo drzewo idą do niego jednym wywołaniem
            drzewa = perm[aktywne, k]
            idx = np.argsort(drzewa, kind="stable")
            probki, drzewa = aktywne[idx], drzewa[idx]
            granice = np.flatnonzero(np.diff(drzewa)) + 1
            for grupa, j in zip(np.split(probki, granice), drzewa[np.r_[0, granice]]):
                y[grupa] = self.trees[j].predict(X[grupa])
            d = y[aktywne] - srednia[aktywne]
            srednia[aktywne] += d / (k + 1)
            S[aktywne] += d * (y[aktywne] - srednia[aktywne])
            stop = S[aktywne] <= self.prog_S[k + 1]
            n_used[aktywne[stop]] = k + 1
            aktywne = aktywne[~stop]
            if len(aktywne) == 0:
                break
        return srednia, n_used

    def predict_one(self, x: ArrayLike) -> tuple[Any, int]:
        x = self._X32(x)
        trees = self.trees
        if self.order == "random":
            trees = [self.trees[j] for j in self.rng.permutation(self.T)]
        prog = self._prog_lista
        srednia, S, t = 0.0, 0.0, 0
        for tree in trees:
            t += 1
            y = float(tree.predict(x)[0])
            d = y - srednia
            srednia += d / t
            S += d * (y - srednia)
            if S <= prog[t]:
                break
        return srednia, t


class IBRegression(SequentialRegression):
    """Regresyjny IB pruning: przedział predykcyjny t-Studenta dla średniej pełnego lasu,
    c[t] = t_{t-1, (1+pi)/2} * sqrt((T - t) / (T * t))."""

    PREFIX = "IBREG"

    def _mnoznik(self) -> FloatArray:
        return mnoznik_studenta(self.T, self.pi, max(self.t_min, 2))


class GleeRegression(SequentialRegression):
    """Regresyjny GLEE: dwustronny przedział normalny dla średniej pełnego lasu,
    c[t] = z_{(1+pi)/2} * rho[t] / sqrt(t), rho to poprawka na populację skończoną.
    t_min=None: minimalna liczba drzew jak w klasyfikacyjnym GLEE (15, 30 albo 45)."""

    PREFIX = "GLEEREG"

    def __init__(self, pi: float = 0.99, delta: float | None = None,
                 delta_std: float | None = None, t_min: int | None = None, fpc: bool = True,
                 name: str | None = None, order: Order = "fixed",
                 seed: int | None = None) -> None:
        self.fpc = fpc
        if t_min is None:                           # przybliżenie normalne jest słabe dla małych t
            t_min = min_glosow(round(1 - pi, 12))
        super().__init__(pi, delta, delta_std, t_min, name, order, seed)

    def _mnoznik(self) -> FloatArray:
        return mnoznik_normalny(self.T, self.pi, self.t_min, self.fpc)
