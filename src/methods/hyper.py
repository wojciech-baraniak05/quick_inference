"""HYPER: instance-based pruning z a priori rozkładu głosów szacowanym z danych
(Soto, Suárez, Martínez-Muñoz, "An urn model for majority voting in classification
ensembles", NIPS 2016). Model i algorytm: implementacja.md, rozdz. 5."""
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import uniform_filter, uniform_filter1d
from scipy.special import gammaln
from sklearn.ensemble import RandomForestClassifier

from generate_table_IB import pi_tag

from .base import (MULTICLASS, ORDER_SUFFIX, BoolArray, FloatArray, IntArray, Multiclass, Order,
                   SequentialPruning)
from .oob import glosy_oob

# największa dopuszczalna liczba komórek (T+1)^(l-1) rozkładu a priori
MAX_CELLS = 2_000_000


def _glosy_niepuste(votes: ArrayLike) -> IntArray:
    votes = np.asarray(votes)
    votes = votes[votes.sum(axis=1) > 0]
    if len(votes) == 0:
        raise ValueError("brak głosów do oszacowania a priori (żadna próbka nie jest out-of-bag)")
    return votes


def sklad_urny(votes: IntArray, T: int) -> IntArray:
    """Głosy (n, l) przeskalowane do T drzew; metoda największych reszt daje sumę dokładnie T."""
    l = votes.shape[1]
    ideal = T * votes / votes.sum(axis=1, keepdims=True)
    sklad = np.floor(ideal).astype(int)
    brakuje = T - sklad.sum(axis=1)                             # 0..l-1 głosów do rozdania
    kolejnosc = np.argsort(sklad - ideal, axis=1, kind="stable")  # największe reszty najpierw
    sklad[np.arange(len(sklad))[:, None], kolejnosc] += np.arange(l) < brakuje[:, None]
    return sklad


def prior_z_glosow(votes: ArrayLike, T: int, smooth: int = 5) -> FloatArray:
    """A priori składu urny z głosów (n, l): tablica (T+1,)^(l-1) indeksowana T_1..T_{l-1}."""
    votes = _glosy_niepuste(votes)
    l = votes.shape[1]
    sklad = sklad_urny(votes, T)

    prior = np.zeros((T + 1,) * (l - 1))
    np.add.at(prior, tuple(sklad[:, :-1].T), 1)
    if smooth and smooth > 1:
        # filtr potrafi zostawić wartości rzędu -1e-20 zamiast 0, a ujemne
        # "prawdopodobieństwo" daje potem nan w całej tablicy progów
        prior = np.clip(uniform_filter(prior, size=smooth, mode="constant"), 0, None)
    prior[np.indices(prior.shape).sum(axis=0) > T] = 0          # składy niemożliwe
    return prior / prior.sum()


def prior_par(votes: ArrayLike, T: int, smooth: int = 5) -> FloatArray:
    """A priori dla pary lider - druga klasa: prior[c, T_c] to rozkład liczby głosów klasy c
    w pojedynku z jej rywalem (T_c + T_rywala = T), z próbek, w których c jest jedną z dwóch
    klas o największej liczbie głosów. Klasa bez takich próbek ma rozkład jednostajny."""
    votes = _glosy_niepuste(votes)
    l = votes.shape[1]
    wiersze = np.arange(len(votes))[:, None]
    # lider i druga klasa każdej próbki, zapisane w kolejności klas
    para = np.sort(np.argsort(-votes, axis=1, kind="stable")[:, :2], axis=1)
    glosy = votes[wiersze, para]
    sklad = sklad_urny(glosy, T)
    # druga klasa bez głosów jest przy l > 2 wybrana przypadkiem, więc nie trafia do histogramu
    licz = (glosy > 0) | (l == 2)

    prior = np.zeros((l, T + 1))
    np.add.at(prior, (para[licz], sklad[licz]), 1)
    if smooth and smooth > 1:
        prior = np.clip(uniform_filter1d(prior, size=smooth, axis=1, mode="constant"), 0, None)
    suma = prior.sum(axis=1, keepdims=True)
    return np.where(suma > 0, prior / np.where(suma > 0, suma, 1), 1 / (T + 1))


def prior_jednostajny(T: int, l: int, para: bool = False) -> FloatArray:
    """Jednostajne a priori w formacie prior_z_glosow (wszystkie składy o sumie T)
    albo, dla para=True, w formacie prior_par."""
    if para:
        return np.full((l, T + 1), 1 / (T + 1))
    prior = np.ones((T + 1,) * (l - 1))
    prior[np.indices(prior.shape).sum(axis=0) > T] = 0
    return prior / prior.sum()


def _lider(votes: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    """Dla votes (n, l): indeks klasy prowadzącej (remis -> niższy indeks), jej głosy
    oraz głosy pozostałych klas (n, l-1) w kolejności klas."""
    l = votes.shape[1]
    wiersze = np.arange(len(votes))
    c = votes.argmax(axis=1)
    kolumny = np.arange(l - 1) + (np.arange(l - 1) >= c[:, None])   # wszystkie poza c
    return c, votes[wiersze, c], votes[wiersze[:, None], kolumny]


def tablica_hyper(prior: FloatArray, pi: float) -> IntArray:
    """prog[c, <głosy pozostałych klas>] = najmniejsza liczba głosów klasy prowadzącej c,
    od której P*(t, T) >= pi."""
    l, T = prior.ndim + 1, prior.shape[0] - 1
    ksztalt = prior.shape
    siatka = np.indices(ksztalt).reshape(l - 1, -1)             # t_1..t_{l-1} każdej komórki
    suma = siatka.sum(axis=0)

    # warstwa T: G[c] = g_c, G[l] = h (przeskalowane; liczy się tylko iloraz g_c / h)
    ostatnia = T - suma
    sklad = np.vstack([siatka, np.maximum(ostatnia, 0)])
    with np.errstate(divide="ignore"):
        log_h = np.log(prior.ravel()) + gammaln(sklad + 1).sum(axis=0)
    log_h[ostatnia < 0] = -np.inf
    h = np.exp(log_h - log_h.max())
    G = np.zeros((l + 1, h.size))
    G[sklad.argmax(axis=0), np.arange(h.size)] = h
    G[l] = h
    G = G.reshape(l + 1, *ksztalt)

    prog = np.zeros((l, *ksztalt), dtype=np.min_scalar_type(T + 1))
    for s in range(T - 1, 0, -1):
        nowe = G.copy()                                         # głos dla ostatniej klasy
        for i in range(1, l):                                   # głos dla klasy i-1
            cel, zrodlo = [slice(None)] * l, [slice(None)] * l
            cel[i], zrodlo[i] = slice(0, -1), slice(1, None)
            nowe[tuple(cel)] += G[tuple(zrodlo)]
        G = nowe.reshape(l + 1, -1)
        G[:, suma > s] = 0                                      # stany spoza warstwy s
        if G[l].max() > 0:
            G /= G[l].max()                                     # ochrona przed nadmiarem

        kom = np.flatnonzero(suma <= s)
        c, top, reszta = _lider(np.column_stack([siatka[:, kom].T, s - suma[kom]]))
        g, h = G[c, kom], G[l, kom]
        pewne = top - reszta.max(axis=1) > T - s                # reszta głosów nic nie zmieni
        zle = ~(pewne | ((h > 0) & (g >= pi * h)))
        np.maximum.at(prog, (c[zle], *reszta[zle].T), top[zle] + 1)
        G = G.reshape(l + 1, *ksztalt)
    return prog


def tablica_par(prior: FloatArray, pi: float) -> IntArray:
    """Dla a priori z prior_par: prog[c, t_2] = najmniejsza liczba głosów lidera c, od której
    przy t_2 głosach drugiej klasy P*(t, T) >= pi w urnie tych dwóch klas."""
    T = prior.shape[1] - 1
    # klasa c jako druga klasa urny binarnej: remis w pełnym zespole to jej przegrana
    prog = np.stack([tablica_hyper(p[::-1], pi)[1] for p in prior])
    # przy remisie w głosach lider nie jest wyznaczony, więc stop wymaga t_1 > t_2
    return np.maximum(prog, np.arange(1, T + 2, dtype=prog.dtype))


def czy_stop_hyper(votes: IntArray, prog: IntArray) -> BoolArray:
    c, top, reszta = _lider(votes)
    return top >= prog[(c, *reszta.T)]


def czy_stop_hyper_para(votes: IntArray, prog: IntArray) -> BoolArray:
    t = np.partition(votes, -2, axis=1)             # dwie ostatnie kolumny: t_2, t_1
    return t[:, -1] >= prog[votes.argmax(axis=1), t[:, -2]]


class Hyper(SequentialPruning):
    """Instance-based pruning z a priori rozkładu głosów szacowanym z głosów out-of-bag.
    multiclass="pair": tablice binarne dla lidera i drugiej klasy, bez limitu liczby klas."""

    PREFIX = "HYPER"
    pi: float
    prior: FloatArray
    prog: IntArray

    def __init__(self, pi: float = 0.99, n_prior: int | None = 20000, smooth: int = 5,
                 name: str | None = None, order: Order = "fixed", seed: int | None = None,
                 multiclass: Multiclass = "full") -> None:
        if not 0 < pi <= 1:
            raise ValueError(f"{type(self).__name__}: pi musi być w (0, 1], a jest {pi}")
        if multiclass not in MULTICLASS:
            raise ValueError(f"{type(self).__name__}: multiclass musi być jednym z {MULTICLASS}, "
                             f"a jest {multiclass!r}")
        self.pi, self.n_prior, self.smooth, self.multiclass = pi, n_prior, smooth, multiclass
        if name is None:
            name = (f"{self.PREFIX}_pi{pi_tag(pi)}" + ("_pair" if multiclass == "pair" else "")
                    + ORDER_SUFFIX[order])
        super().__init__(name, order=order, seed=seed)

    def _prior(self, prior: FloatArray) -> FloatArray:
        """A priori użyte do tablicy progów; podklasa może zmienić oszacowanie z głosów."""
        return prior

    def _prog(self, prog: IntArray) -> IntArray:
        """Tablica progów użyta przy predykcji; podklasa może ją połączyć z inną regułą."""
        return prog

    def fit(self, rf: RandomForestClassifier, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None) -> Self:
        super().fit(rf)
        if X_train is None:
            raise ValueError(f"{self.name}: fit(rf, X_train) wymaga danych uczących lasu "
                             f"(głosy out-of-bag)")
        para = self.multiclass == "pair"
        if not para and (self.T + 1) ** (self.l - 1) > MAX_CELLS:
            raise ValueError(f"{self.name}: T={self.T}, l={self.l} to za dużo stanów "
                             f"((T+1)^(l-1) > {MAX_CELLS}); zmniejsz n_trees albo ustaw "
                             f'multiclass = "pair"')
        votes = glosy_oob(rf, X_train, self.n_prior)
        if para:
            self.prior = self._prior(prior_par(votes, self.T, self.smooth))
            self.prog = self._prog(tablica_par(self.prior, self.pi))
        else:
            self.prior = self._prior(prior_z_glosow(votes, self.T, self.smooth))
            self.prog = self._prog(tablica_hyper(self.prior, self.pi))
        return self

    def _stop_batch(self, votes: IntArray) -> BoolArray:
        if self.multiclass == "pair":
            return czy_stop_hyper_para(votes, self.prog)
        return czy_stop_hyper(votes, self.prog)

    def _stop_one(self, votes: IntArray) -> bool:
        c = int(votes.argmax())
        if self.multiclass == "pair":
            return bool(votes[c] >= self.prog[c, np.partition(votes, -2)[-2]])
        return bool(votes[c] >= self.prog[(c, *np.delete(votes, c))])

    def expected_trees(self, n_sim: int = 2000, seed: int = 0) -> float:
        """Oczekiwana liczba odpytanych drzew na próbkę, Monte Carlo z samego a priori."""
        if self.multiclass == "pair":
            raise ValueError(f"{self.name}: expected_trees wymaga a priori pełnego składu urny "
                             f'(multiclass = "full")')
        rng = np.random.default_rng(seed)
        kom = rng.choice(self.prior.size, size=n_sim, p=self.prior.ravel())
        czesc = np.column_stack(np.unravel_index(kom, self.prior.shape))
        sklady = np.column_stack([czesc, self.T - czesc.sum(axis=1)])
        n_used = np.empty(n_sim, dtype=int)
        for j, sklad in enumerate(sklady):
            glosy = rng.permutation(np.repeat(np.arange(self.l), sklad))
            votes = np.cumsum(glosy[:, None] == np.arange(self.l), axis=0)
            stop = self._stop_batch(votes)
            n_used[j] = stop.argmax() + 1 if stop.any() else self.T
        return float(n_used.mean())
