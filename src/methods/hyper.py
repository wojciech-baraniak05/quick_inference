"""HYPER: instance-based pruning z a priori rozkładu głosów szacowanym z danych
(Soto, Suárez, Martínez-Muñoz, "An urn model for majority voting in classification
ensembles", NIPS 2016). Model i algorytm: implementacja.md, rozdz. 5."""
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import uniform_filter, uniform_filter1d
from scipy.special import gammaln
from sklearn.ensemble import RandomForestClassifier

from .base import (CALIB_SUFFIX, MULTICLASS, ORDER_SUFFIX, BoolArray, Calib, FloatArray, IntArray,
                   Multiclass, Order, SequentialPruning, check_calib, pi_tag)
from .oob import oob_votes
from .validation import val_votes

# największa dopuszczalna liczba komórek (T+1)^(l-1) rozkładu a priori
MAX_CELLS = 2_000_000


def _nonempty_votes(votes: ArrayLike) -> IntArray:
    votes = np.asarray(votes)
    votes = votes[votes.sum(axis=1) > 0]
    if len(votes) == 0:
        raise ValueError("no votes to estimate the prior (no out-of-bag or validation samples)")
    return votes


def urn_composition(votes: IntArray, T: int) -> IntArray:
    """Głosy (n, l) przeskalowane do T drzew; metoda największych reszt daje sumę dokładnie T."""
    l = votes.shape[1]
    ideal = T * votes / votes.sum(axis=1, keepdims=True)
    comp = np.floor(ideal).astype(int)
    n_missing = T - comp.sum(axis=1)                            # 0..l-1 głosów do rozdania
    by_remainder = np.argsort(comp - ideal, axis=1, kind="stable")  # największe reszty najpierw
    comp[np.arange(len(comp))[:, None], by_remainder] += np.arange(l) < n_missing[:, None]
    return comp


def prior_from_votes(votes: ArrayLike, T: int, smooth: int = 5) -> FloatArray:
    """A priori składu urny z głosów (n, l): tablica (T+1,)^(l-1) indeksowana T_1..T_{l-1}."""
    votes = _nonempty_votes(votes)
    l = votes.shape[1]
    comp = urn_composition(votes, T)

    prior = np.zeros((T + 1,) * (l - 1))
    np.add.at(prior, tuple(comp[:, :-1].T), 1)
    if smooth and smooth > 1:
        # filtr potrafi zostawić wartości rzędu -1e-20 zamiast 0, a ujemne
        # "prawdopodobieństwo" daje potem nan w całej tablicy progów
        prior = np.clip(uniform_filter(prior, size=smooth, mode="constant"), 0, None)
    prior[np.indices(prior.shape).sum(axis=0) > T] = 0          # składy niemożliwe
    return prior / prior.sum()


def pair_prior(votes: ArrayLike, T: int, smooth: int = 5) -> FloatArray:
    """A priori dla pary lider - druga klasa: prior[c, T_c] to rozkład liczby głosów klasy c
    w pojedynku z jej rywalem (T_c + T_rywala = T), z próbek, w których c jest jedną z dwóch
    klas o największej liczbie głosów. Klasa bez takich próbek ma rozkład jednostajny."""
    votes = _nonempty_votes(votes)
    l = votes.shape[1]
    rows = np.arange(len(votes))[:, None]
    # lider i druga klasa każdej próbki, zapisane w kolejności klas
    pair = np.sort(np.argsort(-votes, axis=1, kind="stable")[:, :2], axis=1)
    pair_votes = votes[rows, pair]
    comp = urn_composition(pair_votes, T)
    # druga klasa bez głosów jest przy l > 2 wybrana przypadkiem, więc nie trafia do histogramu
    counted = (pair_votes > 0) | (l == 2)

    prior = np.zeros((l, T + 1))
    np.add.at(prior, (pair[counted], comp[counted]), 1)
    if smooth and smooth > 1:
        prior = np.clip(uniform_filter1d(prior, size=smooth, axis=1, mode="constant"), 0, None)
    total = prior.sum(axis=1, keepdims=True)
    return np.where(total > 0, prior / np.where(total > 0, total, 1), 1 / (T + 1))


def uniform_prior(T: int, l: int, pair: bool = False) -> FloatArray:
    """Jednostajne a priori w formacie prior_from_votes (wszystkie składy o sumie T)
    albo, dla pair=True, w formacie pair_prior."""
    if pair:
        return np.full((l, T + 1), 1 / (T + 1))
    prior = np.ones((T + 1,) * (l - 1))
    prior[np.indices(prior.shape).sum(axis=0) > T] = 0
    return prior / prior.sum()


def _leader(votes: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    """Dla votes (n, l): indeks klasy prowadzącej (remis -> niższy indeks), jej głosy
    oraz głosy pozostałych klas (n, l-1) w kolejności klas."""
    l = votes.shape[1]
    rows = np.arange(len(votes))
    c = votes.argmax(axis=1)
    cols = np.arange(l - 1) + (np.arange(l - 1) >= c[:, None])      # wszystkie poza c
    return c, votes[rows, c], votes[rows[:, None], cols]


def hyper_table(prior: FloatArray, pi: float) -> IntArray:
    """threshold[c, <głosy pozostałych klas>] = najmniejsza liczba głosów klasy prowadzącej c,
    od której P*(t, T) >= pi."""
    l, T = prior.ndim + 1, prior.shape[0] - 1
    shape = prior.shape
    grid = np.indices(shape).reshape(l - 1, -1)                 # t_1..t_{l-1} każdej komórki
    total = grid.sum(axis=0)

    # warstwa T: G[c] = g_c, G[l] = h (przeskalowane; liczy się tylko iloraz g_c / h)
    last = T - total
    comp = np.vstack([grid, np.maximum(last, 0)])
    with np.errstate(divide="ignore"):
        log_h = np.log(prior.ravel()) + gammaln(comp + 1).sum(axis=0)
    log_h[last < 0] = -np.inf
    h = np.exp(log_h - log_h.max())
    G = np.zeros((l + 1, h.size))
    G[comp.argmax(axis=0), np.arange(h.size)] = h
    G[l] = h
    G = G.reshape(l + 1, *shape)

    threshold = np.zeros((l, *shape), dtype=np.min_scalar_type(T + 1))
    for s in range(T - 1, 0, -1):
        new_G = G.copy()                                        # głos dla ostatniej klasy
        for i in range(1, l):                                   # głos dla klasy i-1
            dst, src = [slice(None)] * l, [slice(None)] * l
            dst[i], src[i] = slice(0, -1), slice(1, None)
            new_G[tuple(dst)] += G[tuple(src)]
        G = new_G.reshape(l + 1, -1)
        G[:, total > s] = 0                                     # stany spoza warstwy s
        if G[l].max() > 0:
            G /= G[l].max()                                     # ochrona przed nadmiarem

        cells = np.flatnonzero(total <= s)
        c, top, rest = _leader(np.column_stack([grid[:, cells].T, s - total[cells]]))
        g, h = G[c, cells], G[l, cells]
        decided = top - rest.max(axis=1) > T - s                # reszta głosów nic nie zmieni
        undecided = ~(decided | ((h > 0) & (g >= pi * h)))
        np.maximum.at(threshold, (c[undecided], *rest[undecided].T), top[undecided] + 1)
        G = G.reshape(l + 1, *shape)
    return threshold


def pair_table(prior: FloatArray, pi: float) -> IntArray:
    """Dla a priori z pair_prior: threshold[c, t_2] = najmniejsza liczba głosów lidera c, od której
    przy t_2 głosach drugiej klasy P*(t, T) >= pi w urnie tych dwóch klas."""
    T = prior.shape[1] - 1
    # klasa c jako druga klasa urny binarnej: remis w pełnym zespole to jej przegrana
    threshold = np.stack([hyper_table(p[::-1], pi)[1] for p in prior])
    # przy remisie w głosach lider nie jest wyznaczony, więc stop wymaga t_1 > t_2
    return np.maximum(threshold, np.arange(1, T + 2, dtype=threshold.dtype))


def stop_hyper(votes: IntArray, threshold: IntArray) -> BoolArray:
    c, top, rest = _leader(votes)
    return top >= threshold[(c, *rest.T)]


def stop_hyper_pair(votes: IntArray, threshold: IntArray) -> BoolArray:
    t = np.partition(votes, -2, axis=1)             # dwie ostatnie kolumny: t_2, t_1
    return t[:, -1] >= threshold[votes.argmax(axis=1), t[:, -2]]


class Hyper(SequentialPruning):
    """Instance-based pruning z a priori rozkładu głosów szacowanym z głosów out-of-bag
    (calib="val": z głosów na zbiorze walidacyjnym).
    multiclass="pair": tablice binarne dla lidera i drugiej klasy, bez limitu liczby klas."""

    PREFIX = "HYPER"
    pi: float
    prior: FloatArray
    threshold: IntArray

    def __init__(self, pi: float = 0.99, n_prior: int | None = 20000, smooth: int = 5,
                 name: str | None = None, order: Order = "fixed", seed: int | None = None,
                 multiclass: Multiclass = "full", calib: Calib = "oob") -> None:
        if not 0 < pi <= 1:
            raise ValueError(f"{type(self).__name__}: pi must be in (0, 1], got {pi}")
        if multiclass not in MULTICLASS:
            raise ValueError(f"{type(self).__name__}: multiclass must be one of {MULTICLASS}, "
                             f"got {multiclass!r}")
        check_calib(type(self).__name__, calib)
        self.pi, self.n_prior, self.smooth, self.multiclass = pi, n_prior, smooth, multiclass
        self.calib = calib
        if name is None:
            name = (f"{self.PREFIX}_pi{pi_tag(pi)}" + ("_pair" if multiclass == "pair" else "")
                    + CALIB_SUFFIX[calib] + ORDER_SUFFIX.get(order, ""))
        super().__init__(name, order=order, seed=seed)

    def _prior(self, prior: FloatArray) -> FloatArray:
        """A priori użyte do tablicy progów; podklasa może zmienić oszacowanie z głosów."""
        return prior

    def _threshold(self, threshold: IntArray) -> IntArray:
        """Tablica progów użyta przy predykcji; podklasa może ją połączyć z inną regułą."""
        return threshold

    def fit(self, rf: RandomForestClassifier, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None, X_val: ArrayLike | None = None,
            y_val: ArrayLike | None = None) -> Self:
        super().fit(rf)
        if self.calib == "val" and X_val is None:
            raise ValueError(f"{self.name}: calib='val' needs a validation set in "
                             f"fit(rf, X_train, y_train, X_val) (benchmark.val_size > 0)")
        if self.calib == "oob" and X_train is None:
            raise ValueError(f"{self.name}: fit(rf, X_train) needs the forest's training data "
                             f"(out-of-bag votes)")
        pair = self.multiclass == "pair"
        if not pair and (self.T + 1) ** (self.l - 1) > MAX_CELLS:
            raise ValueError(f"{self.name}: T={self.T}, l={self.l} gives too many states "
                             f"((T+1)^(l-1) > {MAX_CELLS}); reduce n_trees or set "
                             f'multiclass = "pair"')
        if self.calib == "val":
            votes = val_votes(rf, X_val, self.n_prior)
        else:
            votes = oob_votes(rf, X_train, self.n_prior)
        if pair:
            self.prior = self._prior(pair_prior(votes, self.T, self.smooth))
            self.threshold = self._threshold(pair_table(self.prior, self.pi))
        else:
            self.prior = self._prior(prior_from_votes(votes, self.T, self.smooth))
            self.threshold = self._threshold(hyper_table(self.prior, self.pi))
        return self

    def _stop_batch(self, votes: IntArray) -> BoolArray:
        if self.multiclass == "pair":
            return stop_hyper_pair(votes, self.threshold)
        return stop_hyper(votes, self.threshold)

    def _stop_one(self, votes: IntArray) -> bool:
        c = int(votes.argmax())
        if self.multiclass == "pair":
            return bool(votes[c] >= self.threshold[c, np.partition(votes, -2)[-2]])
        return bool(votes[c] >= self.threshold[(c, *np.delete(votes, c))])

    def expected_trees(self, n_sim: int = 2000, seed: int = 0) -> float:
        """Oczekiwana liczba odpytanych drzew na próbkę, Monte Carlo z samego a priori."""
        if self.multiclass == "pair":
            raise ValueError(f"{self.name}: expected_trees needs the full urn prior "
                             f'(multiclass = "full")')
        rng = np.random.default_rng(seed)
        cells = rng.choice(self.prior.size, size=n_sim, p=self.prior.ravel())
        partial = np.column_stack(np.unravel_index(cells, self.prior.shape))
        comps = np.column_stack([partial, self.T - partial.sum(axis=1)])
        n_used = np.empty(n_sim, dtype=int)
        for j, comp in enumerate(comps):
            ballots = rng.permutation(np.repeat(np.arange(self.l), comp))
            votes = np.cumsum(ballots[:, None] == np.arange(self.l), axis=0)
            stop = self._stop_batch(votes)
            n_used[j] = stop.argmax() + 1 if stop.any() else self.T
        return float(n_used.mean())
