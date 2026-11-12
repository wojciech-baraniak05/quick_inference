"""HYPER: instance-based pruning z a priori rozkładu głosów szacowanym z danych
(Soto, Suárez, Martínez-Muñoz, "An urn model for majority voting in classification
ensembles", NIPS 2016). Model i algorytm: implementacja.md, rozdz. 5."""
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import uniform_filter
from scipy.special import gammaln
from sklearn.ensemble import RandomForestClassifier

from generate_table_IB import pi_tag

from .base import BoolArray, FloatArray, IntArray, SequentialPruning
from .oob import glosy_oob

# największa dopuszczalna liczba komórek (T+1)^(l-1) rozkładu a priori
MAX_CELLS = 2_000_000


def prior_z_glosow(votes: ArrayLike, T: int, smooth: int = 5) -> FloatArray:
    """A priori składu urny z głosów (n, l): tablica (T+1,)^(l-1) indeksowana T_1..T_{l-1}."""
    votes = np.asarray(votes)
    votes = votes[votes.sum(axis=1) > 0]
    if len(votes) == 0:
        raise ValueError("brak głosów do oszacowania a priori (żadna próbka nie jest out-of-bag)")
    l = votes.shape[1]
    # głosy przeskalowane do T drzew; metoda największych reszt daje sumę dokładnie T
    ideal = T * votes / votes.sum(axis=1, keepdims=True)
    sklad = np.floor(ideal).astype(int)
    brakuje = T - sklad.sum(axis=1)                             # 0..l-1 głosów do rozdania
    kolejnosc = np.argsort(sklad - ideal, axis=1, kind="stable")  # największe reszty najpierw
    sklad[np.arange(len(sklad))[:, None], kolejnosc] += np.arange(l) < brakuje[:, None]

    prior = np.zeros((T + 1,) * (l - 1))
    np.add.at(prior, tuple(sklad[:, :-1].T), 1)
    if smooth and smooth > 1:
        # filtr potrafi zostawić wartości rzędu -1e-20 zamiast 0, a ujemne
        # "prawdopodobieństwo" daje potem nan w całej tablicy progów
        prior = np.clip(uniform_filter(prior, size=smooth, mode="constant"), 0, None)
    prior[np.indices(prior.shape).sum(axis=0) > T] = 0          # składy niemożliwe
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


def czy_stop_hyper(votes: IntArray, prog: IntArray) -> BoolArray:
    c, top, reszta = _lider(votes)
    return top >= prog[(c, *reszta.T)]


class Hyper(SequentialPruning):
    """Instance-based pruning z a priori rozkładu głosów szacowanym z głosów out-of-bag."""

    pi: float
    prior: FloatArray
    prog: IntArray

    def __init__(self, pi: float = 0.99, n_prior: int | None = 20000, smooth: int = 5,
                 name: str | None = None) -> None:
        if not 0 < pi <= 1:
            raise ValueError(f"Hyper: pi musi być w (0, 1], a jest {pi}")
        self.pi, self.n_prior, self.smooth = pi, n_prior, smooth
        super().__init__(name or f"HYPER_pi{pi_tag(pi)}")

    def fit(self, rf: RandomForestClassifier, X_train: ArrayLike | None = None,
            y_train: ArrayLike | None = None) -> Self:
        super().fit(rf)
        if X_train is None:
            raise ValueError(f"{self.name}: fit(rf, X_train) wymaga danych uczących lasu "
                             f"(głosy out-of-bag)")
        if (self.T + 1) ** (self.l - 1) > MAX_CELLS:
            raise ValueError(f"{self.name}: T={self.T}, l={self.l} to za dużo stanów "
                             f"((T+1)^(l-1) > {MAX_CELLS}); zmniejsz n_trees")
        self.prior = prior_z_glosow(glosy_oob(rf, X_train, self.n_prior), self.T, self.smooth)
        self.prog = tablica_hyper(self.prior, self.pi)
        return self

    def _stop_batch(self, votes: IntArray) -> BoolArray:
        return czy_stop_hyper(votes, self.prog)

    def _stop_one(self, votes: IntArray) -> bool:
        c = int(votes.argmax())
        return bool(votes[c] >= self.prog[(c, *np.delete(votes, c))])

    def expected_trees(self, n_sim: int = 2000, seed: int = 0) -> float:
        """Oczekiwana liczba odpytanych drzew na próbkę, Monte Carlo z samego a priori."""
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
