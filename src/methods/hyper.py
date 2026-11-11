"""HYPER: instance-based pruning z nieparametrycznym a priori rozkładu głosów
(Soto, Suárez, Martínez-Muñoz, "An urn model for majority voting in
classification ensembles", NIPS 2016).

Las dla jednej próbki to urna z T kulami (drzewa) w l kolorach (klasy); skład urny
T = (T_1..T_l) jest nieznany, a odpytywanie drzew to losowanie bez zwracania.
Po zebraniu głosów t = (t_1..t_l) rozkład a posteriori składu urny (wzór (2)) to

    P(T | t)  ~  C(T_1, t_1) * ... * C(T_l, t_l) * P(T)

a P*(t, T) (wzór (3)) to suma P(T | t) po składach, w których wygrywa ta sama klasa
co w t. Głosowanie przerywamy, gdy P*(t, T) >= pi.

IB pruning (SIBA) to ten sam model z jednostajnym P(T). Tutaj P(T) jest szacowane
z głosów out-of-bag na zbiorze uczącym, więc tablica progów zależy od danych
i jest liczona w fit(), a nie czytana z pliku.
"""
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.special import gammaln

from generate_table_IB import pi_tag

from .base import SequentialPruning

# największa dopuszczalna liczba komórek (T+1)^(l-1) rozkładu a priori
MAX_CELLS = 2_000_000


def glosy_oob(rf, X_train, n_prior=None, seed=0):
    """Głosy out-of-bag: votes[n, i] = ile drzew, które NIE widziały próbki n
    przy uczeniu, głosuje na klasę i. X_train to dane, na których uczono rf.

    n_prior ogranicza liczbę (losowo wybranych) próbek uczących; None = wszystkie.
    """
    X_train = np.asarray(X_train)
    n = len(X_train)
    if not rf.bootstrap:
        raise ValueError("głosy out-of-bag wymagają lasu z bootstrap=True")
    if getattr(rf, "_n_samples", n) != n:
        raise ValueError(f"X_train ma {n} wierszy, a las uczono na {rf._n_samples}")
    if n_prior is None or n_prior >= n:
        wiersze = np.arange(n)
    else:
        wiersze = np.sort(np.random.default_rng(seed).choice(n, n_prior, replace=False))
    X = X_train[wiersze]

    # indeksy bootstrapu każdego drzewa; generator sklearn nie trzyma ich naraz w pamięci
    if hasattr(rf, "_get_estimators_indices"):
        indeksy = rf._get_estimators_indices()
    else:
        indeksy = rf.estimators_samples_
    votes = np.zeros((len(wiersze), len(rf.classes_)), dtype=int)
    for tree, w_probie in zip(rf.estimators_, indeksy):
        oob = np.flatnonzero(np.bincount(w_probie, minlength=n)[wiersze] == 0)
        votes[oob, tree.predict(X[oob]).astype(int)] += 1
    return votes


def prior_z_glosow(votes, T, smooth=5):
    """Nieparametryczne a priori P(T) z głosów (np. out-of-bag), rozdz. 2.2.

    Głosy każdej próbki są przeskalowane do zespołu T drzew, T_i ~ T * votes_i / sum(votes)
    (zaokrąglenie metodą największych reszt, żeby suma wyszła dokładnie T), a histogram
    składów jest wygładzony oknem przesuwnym o szerokości smooth.

    Zwraca tablicę o kształcie (T+1,)^(l-1): prior[T_1, ..., T_{l-1}], gdzie
    T_l = T - (T_1 + ... + T_{l-1}); komórki o sumie > T mają 0.
    """
    votes = np.asarray(votes)
    votes = votes[votes.sum(axis=1) > 0]
    if len(votes) == 0:
        raise ValueError("brak głosów do oszacowania a priori (żadna próbka nie jest out-of-bag)")
    l = votes.shape[1]
    ideal = T * votes / votes.sum(axis=1, keepdims=True)
    sklad = np.floor(ideal).astype(int)
    brakuje = T - sklad.sum(axis=1)                             # 0..l-1 głosów do rozdania
    kolejnosc = np.argsort(sklad - ideal, axis=1, kind="stable")  # największe reszty najpierw
    sklad[np.arange(len(sklad))[:, None], kolejnosc] += np.arange(l) < brakuje[:, None]

    prior = np.zeros((T + 1,) * (l - 1))
    np.add.at(prior, tuple(sklad[:, :-1].T), 1)
    if smooth and smooth > 1:
        prior = uniform_filter(prior, size=smooth, mode="constant")
    prior[np.indices(prior.shape).sum(axis=0) > T] = 0          # składy niemożliwe
    return prior / prior.sum()


def _lider(votes):
    """Dla votes (n, l): indeks klasy prowadzącej (remis -> niższy indeks, jak w argmax
    pełnego głosowania), jej liczba głosów oraz głosy pozostałych klas (n, l-1)
    w kolejności klas."""
    l = votes.shape[1]
    wiersze = np.arange(len(votes))
    c = votes.argmax(axis=1)
    kolumny = np.arange(l - 1) + (np.arange(l - 1) >= c[:, None])   # wszystkie poza c
    return c, votes[wiersze, c], votes[wiersze[:, None], kolumny]


def tablica_hyper(prior, pi):
    """Tablica progów HYPER dla a priori prior (patrz prior_z_glosow) i poziomu pi.

        prog[c, <głosy pozostałych klas w kolejności klas>] = najmniejsza liczba głosów
        klasy prowadzącej c, od której P*(t, T) >= pi

    Klasy nie są wymienne (a priori nie jest symetryczne), więc każda klasa
    prowadząca ma własną tablicę. Jeśli P* nie jest monotoniczne, próg leży za
    ostatnim stanem z P* < pi, więc każde przerwanie spełnia P* >= pi.

    Liczenie: niech h(t) = suma po T z P(T) * P(konkretna kolejność głosów o liczbach t | T),
    a g_c(t) to ta sama suma tylko po składach T, w których wygrywa klasa c. Wtedy
    P*(t, T) = g_c(t) / h(t) dla c = klasa prowadząca w t, a obie funkcje spełniają

        f(t) = suma_i f(t + e_i)        (następny głos trafia do którejś klasy)

    z warunkiem brzegowym f(T) = P(T) / (T! / (T_1! ... T_l!)) dla kompletu głosów.
    Idziemy więc warstwami od T głosów w dół; stanów jest C(T + l, l).
    """
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


def czy_stop_hyper(votes, prog):
    """Maska stopu dla votes o kształcie (n_próbek, l) i tablicy z tablica_hyper."""
    c, top, reszta = _lider(votes)
    return top >= prog[(c, *reszta.T)]


class Hyper(SequentialPruning):
    """Instance-based pruning z a priori rozkładu głosów szacowanym z danych (HYPER).

        Hyper(pi=0.99)    poziom ufności: przerwij, gdy P*(t, T) >= pi
        n_prior           ile próbek uczących użyć do głosów out-of-bag (None = wszystkie)
        smooth            szerokość okna wygładzającego a priori (artykuł: 5; 0 = bez)

    fit(rf, X_train) wymaga danych, na których uczono las (bootstrap=True).
    Po fit: prior (rozkład a priori składu urny) i prog (tablica progów).
    """

    def __init__(self, pi=0.99, n_prior=20000, smooth=5, name=None):
        if not 0 < pi <= 1:
            raise ValueError(f"Hyper: pi musi być w (0, 1], a jest {pi}")
        self.pi, self.n_prior, self.smooth = pi, n_prior, smooth
        super().__init__(name or f"HYPER_pi{pi_tag(pi)}")

    def fit(self, rf, X_train=None):
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

    def _stop_batch(self, votes):
        return czy_stop_hyper(votes, self.prog)

    def _stop_one(self, votes):
        c = int(votes.argmax())
        return votes[c] >= self.prog[(c, *np.delete(votes, c))]

    def expected_trees(self, n_sim=2000, seed=0):
        """Oczekiwana liczba odpytanych drzew na próbkę, oszacowana metodą Monte Carlo
        tylko z a priori i tablicy progów (bez danych testowych): losujemy skład urny
        z prior, ustawiamy głosy w losowej kolejności i liczymy je aż do stopu."""
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
        return n_used.mean()
