"""Wczesny stop w regresji: metody odniesienia, wspólna pętla oraz regresyjne warianty
IB pruning i GLEE. Opis: implementacja.md, rozdz. 11."""
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri, stdtrit
from sklearn.ensemble import RandomForestRegressor

from .base import (ORDER_SUFFIX, FastRegTree, Features, FloatArray, InferenceMethod, Order,
                   Prediction, SequentialPruning, Task, fast_reg_trees, pi_tag)
from .glee import alpha_from_pi, default_min_votes


class RegressionMethod(InferenceMethod):
    """Bazowa klasa metod dla lasu regresyjnego: predykcją jest średnia wyjść drzew."""

    task: Task = "regression"
    rf: RandomForestRegressor
    trees: list[FastRegTree]

    def fit(self, rf: RandomForestRegressor, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None, X_val: ArrayLike | None = None,
            y_val: ArrayLike | None = None) -> Self:
        if not isinstance(rf, RandomForestRegressor):
            raise ValueError(f"{self.name}: regression method, "
                             f"but the forest is {type(rf).__name__}")
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
        total = np.zeros(len(X))
        for tree in self.trees:
            total += tree.predict(X)
        return total / self.T, np.full(len(X), self.T)

    def predict_one(self, x: ArrayLike) -> tuple[Any, int]:
        x = self._X32(x)
        total = 0.0
        for tree in self.trees:
            total += tree.predict(x)[0]
        return total / self.T, self.T


def fpc_correction(T: int) -> FloatArray:
    """rho[t] = sqrt((T - t) / (T - 1)) dla t > 0.05 * T, inaczej 1 (t = 0..T)."""
    t = np.arange(T + 1)
    return np.where(t > 0.05 * T, np.sqrt((T - t) / max(T - 1, 1)), 1.0)


def student_multiplier(T: int, pi: float, t_min: int = 2, nu0: float = 0.0) -> FloatArray:
    """c[t] = q_{nu0 + t - 1, (1 + pi) / 2} * sqrt((T - t) / (T * t)) dla t = 0..T; inf tam,
    gdzie po t zapytaniach nie wolno przerwać (t < t_min albo brak stopni swobody)."""
    t = np.arange(T + 1)
    df = nu0 + t - 1
    c = np.full(T + 1, np.inf)
    ok = (t >= max(t_min, 1)) & (df > 0)
    with np.errstate(invalid="ignore"):             # pi = 1: inf * 0 dla t = T
        c[ok] = stdtrit(df[ok], (1 + pi) / 2) * np.sqrt((T - t[ok]) / (T * t[ok]))
    return c


def normal_multiplier(T: int, pi: float, t_min: int = 2, fpc: bool = True) -> FloatArray:
    """c[t] = z_{(1 + pi) / 2} * rho[t] / sqrt(t) dla t = 0..T; inf dla t < t_min."""
    t = np.arange(T + 1)
    c = np.full(T + 1, np.inf)
    ok = t >= max(t_min, 2)                         # s_t wymaga co najmniej dwóch wyjść
    rho = fpc_correction(T) if fpc else np.ones(T + 1)
    with np.errstate(invalid="ignore"):             # pi = 1: inf * 0 dla t = T
        c[ok] = ndtri((1 + pi) / 2) * rho[ok] / np.sqrt(t[ok])
    return c


def sum_sq_threshold(c: FloatArray, delta: float, nu0: float = 0.0,
                        sigma0_2: float = 0.0) -> FloatArray:
    """Warunek c[t] * sigma_t <= delta zapisany dla sumy kwadratów odchyleń S_t wyjść:
    stop, gdy S_t <= threshold[t], gdzie sigma_t^2 = (nu0 * sigma0_2 + S_t) / (nu0 + t - 1).
    threshold[t] = -1: po t zapytaniach nie wolno przerwać; threshold[T] = inf."""
    T = len(c) - 1
    t = np.arange(T + 1)
    threshold = np.full(T + 1, -1.0)
    ok = np.isfinite(c) & (c > 0)
    threshold[ok] = (delta / c[ok]) ** 2 * (nu0 + t[ok] - 1) - nu0 * sigma0_2
    threshold[c == 0] = np.inf
    threshold[T] = np.inf                           # po wszystkich drzewach zawsze koniec
    return threshold


class SequentialRegression(RegressionMethod):
    """Wczesny stop w regresji: drzewa odpowiadają po kolei, a próbka kończy, gdy
    multiplier[t] * rozrzut wyjść <= delta. Podklasa podaje _multiplier()."""

    ORDERS = SequentialPruning.ORDERS
    PREFIX = "REG"
    pi: float
    delta: float
    nu0: float = 0.0            # a priori wariancji wyjść: liczba pseudoobserwacji
    sigma0_2: float = 0.0       # i ich wariancja
    multiplier: FloatArray      # c[t], t = 0..T
    threshold_S: FloatArray     # próg sumy kwadratów odchyleń, t = 0..T
    rng: np.random.Generator

    def __init__(self, pi: float = 0.99, delta: float | None = None,
                 delta_std: float | None = None, t_min: int = 2, name: str | None = None,
                 order: Order = "fixed", seed: int | None = None) -> None:
        cls_name = type(self).__name__
        if not 0 < pi <= 1:
            raise ValueError(f"{cls_name}: pi must be in (0, 1], got {pi}")
        if (delta is None) == (delta_std is None):
            raise ValueError(f"{cls_name}: give exactly one of delta (in target units) "
                             f"and delta_std (in standard deviations of y_train)")
        tolerance = delta if delta is not None else delta_std
        assert tolerance is not None
        if tolerance <= 0:
            raise ValueError(f"{cls_name}: tolerance must be positive, got {tolerance}")
        if t_min < 1:
            raise ValueError(f"{cls_name}: t_min must be >= 1, got {t_min}")
        if order not in self.ORDERS:
            raise ValueError(f"{cls_name}: order must be one of {self.ORDERS}, got {order!r}")
        self.pi, self._delta, self.delta_std, self.t_min = pi, delta, delta_std, t_min
        self.order, self.seed = order, seed
        if name is None:
            tol = f"d{delta:g}" if delta is not None else f"ds{delta_std:g}"
            name = f"{self.PREFIX}_pi{pi_tag(pi)}_{tol}{ORDER_SUFFIX.get(order, '')}"
        super().__init__(name)

    def _multiplier(self) -> FloatArray:
        """Tablica c[t] dla t = 0..T; inf tam, gdzie po t zapytaniach nie wolno przerwać."""
        raise NotImplementedError

    def fit(self, rf: RandomForestRegressor, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None, X_val: ArrayLike | None = None,
            y_val: ArrayLike | None = None) -> Self:
        super().fit(rf)
        if self._delta is not None:
            self.delta = self._delta
        else:
            if y_train is None:
                raise ValueError(f"{self.name}: delta_std needs y_train "
                                 f"in fit(rf, X_train, y_train)")
            assert self.delta_std is not None
            self.delta = self.delta_std * float(np.std(np.asarray(y_train, dtype=float)))
            if self.delta <= 0:
                raise ValueError(f"{self.name}: y_train has zero standard deviation, "
                                 f"use delta instead of delta_std")
        self.rng = np.random.default_rng(self.seed)
        if self.order == "shuffled":                # jedna permutacja lasu dla wszystkich próbek
            self.trees = [self.trees[j] for j in self.rng.permutation(self.T)]
        self.multiplier = self._multiplier()
        self.threshold_S = sum_sq_threshold(self.multiplier, self.delta, self.nu0, self.sigma0_2)
        self._threshold_list: list[float] = self.threshold_S.tolist()
        return self

    def predict_batch(self, X: ArrayLike) -> Prediction:
        X = self._X32(X)
        if self.order == "random":
            return self._predict_batch_random(X)
        y_pred = np.empty(len(X))
        n_used = np.full(len(X), self.T)
        active = np.arange(len(X))                  # próbki jeszcze niezdecydowane
        X_active = X                                # ich cechy, średnie i sumy kwadratów
        mean = np.zeros(len(X))
        S = np.zeros(len(X))
        for k, tree in enumerate(self.trees):
            y = tree.predict(X_active)
            d = y - mean
            mean += d / (k + 1)
            S += d * (y - mean)
            stop = S <= self.threshold_S[k + 1]
            if stop.any():
                y_pred[active[stop]] = mean[stop]
                n_used[active[stop]] = k + 1
                active = active[~stop]
                if len(active) == 0:
                    break
                X_active, mean, S = X_active[~stop], mean[~stop], S[~stop]
        return y_pred, n_used

    def _predict_batch_random(self, X: Features) -> Prediction:
        n = len(X)
        n_used = np.full(n, self.T)
        mean = np.zeros(n)
        S = np.zeros(n)
        y = np.empty(n)
        # perm[i, k] = drzewo, które próbka i pyta w kroku k
        perm = np.tile(np.arange(self.T, dtype=np.min_scalar_type(self.T)), (n, 1))
        perm = self.rng.permuted(perm, axis=1)
        active = np.arange(n)
        for k in range(self.T):
            # próbki pytające w tym kroku to samo drzewo idą do niego jednym wywołaniem
            tree_ids = perm[active, k]
            idx = np.argsort(tree_ids, kind="stable")
            samples, tree_ids = active[idx], tree_ids[idx]
            bounds = np.flatnonzero(np.diff(tree_ids)) + 1
            for group, j in zip(np.split(samples, bounds), tree_ids[np.r_[0, bounds]]):
                y[group] = self.trees[j].predict(X[group])
            d = y[active] - mean[active]
            mean[active] += d / (k + 1)
            S[active] += d * (y[active] - mean[active])
            stop = S[active] <= self.threshold_S[k + 1]
            n_used[active[stop]] = k + 1
            active = active[~stop]
            if len(active) == 0:
                break
        return mean, n_used

    def predict_one(self, x: ArrayLike) -> tuple[Any, int]:
        x = self._X32(x)
        trees = self.trees
        if self.order == "random":
            trees = [self.trees[j] for j in self.rng.permutation(self.T)]
        threshold = self._threshold_list
        mean, S, t = 0.0, 0.0, 0
        for tree in trees:
            t += 1
            y = float(tree.predict(x)[0])
            d = y - mean
            mean += d / t
            S += d * (y - mean)
            if S <= threshold[t]:
                break
        return mean, t


class IBRegression(SequentialRegression):
    """Regresyjny IB pruning: przedział predykcyjny t-Studenta dla średniej pełnego lasu,
    c[t] = t_{t-1, (1+pi)/2} * sqrt((T - t) / (T * t))."""

    PREFIX = "IBREG"

    def _multiplier(self) -> FloatArray:
        return student_multiplier(self.T, self.pi, max(self.t_min, 2))


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
            t_min = default_min_votes(alpha_from_pi(pi))
        super().__init__(pi, delta, delta_std, t_min, name, order, seed)

    def _multiplier(self) -> FloatArray:
        return normal_multiplier(self.T, self.pi, self.t_min, self.fpc)
