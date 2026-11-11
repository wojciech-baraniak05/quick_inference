"""GLEE: Gaussian Lazy Ensemble Evaluation, czyli reguła wczesnego stopu z artykułu
Basilico, Munson, Kolda, Dixon, Kegelmeyer, "COMET: A Recipe for Learning and Using
Large Ensembles on Massive Data" (ICDM 2011, arXiv:1103.2068), rozdz. II-C.

Głos drzewa to zmienna Bernoulliego, więc odsetek głosów klasy prowadzącej p^ po n
głosach ma w przybliżeniu rozkład normalny. Głosowanie przerywamy, gdy dolna granica
jednostronnego przedziału ufności (1 - alfa) dla p leży powyżej 0.5:

    p^ - rho * delta > 0.5
    delta = z_alfa * sqrt(p^ * (1 - p^) / n)
    rho   = sqrt((m - n) / (m - 1))  jeśli n > 0.05 * m,  inaczej 1

gdzie m to liczba drzew, a rho to poprawka na populację skończoną (FPC): drzewa są
losowane bez zwracania, więc niepewność maleje, gdy zostaje ich mało. Przybliżenie
normalne wymaga minimalnej liczby głosów (min_votes), zanim wolno przerwać.

W artykule alfa to największa dopuszczalna częstość niezgodności z pełnym głosowaniem;
tutaj podaje się pi = 1 - alfa, tak jak w IB pruning i HYPER.

Dla wielu klas (przypis 1 w artykule) klasa prowadząca jest traktowana jak klasa 1,
a druga w kolejności jak klasa 0: p^ = t_1 / (t_1 + t_2), n = t_1 + t_2.

Z COMET zaimplementowana jest tylko ta reguła. Uczenie (MapReduce, IVoting) dotyczy
budowy zespołu, a nie inferencji na gotowym lesie.
"""
import numpy as np
from scipy.special import ndtri

from generate_table_IB import pi_tag

from .base import SequentialPruning


def min_glosow(alfa):
    """Minimalna liczba głosów przed stopem zalecana w artykule:
    15, 30 i 45 dla alfa = 1e-2, 1e-3 i 1e-4 (dla pośrednich alfa: wartość ostrzejsza)."""
    if alfa >= 1e-2:
        return 15
    return 30 if alfa >= 1e-3 else 45


def tablica_glee(T, alfa, min_votes=None, fpc=True, one_sided=True):
    """Tablica progów GLEE dla zespołu T drzew:

        prog[t_2] = najmniejsze t_1, przy którym p^ - rho * delta > 0.5
                    (T + 1, jeśli przy tym t_2 nie da się przerwać)

    gdzie t_1 >= t_2 to głosy klasy prowadzącej i drugiej w kolejności. Przy ustalonym
    t_2 dolna granica przedziału rośnie z t_1, więc próg wystarcza do decyzji.

        fpc=False        bez poprawki na populację skończoną (warianty G1, G2 z artykułu)
        one_sided=False  przedział dwustronny, z_{alfa/2} (warianty G2, G2-FPC)

    Domyślne ustawienia to G1-FPC, czyli reguła nazywana w artykule GLEE.
    """
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


def czy_stop_glee(votes, prog):
    """Maska stopu dla votes o kształcie (n_próbek, l) i tablicy z tablica_glee."""
    t = np.partition(votes, -2, axis=1)             # dwie ostatnie kolumny: t_2, t_1
    return t[:, -1] >= prog[t[:, -2]]


class Glee(SequentialPruning):
    """Wczesny stop z gaussowskim przedziałem ufności (GLEE z artykułu COMET).

        Glee(pi=0.99)   przerwij, gdy z ufnością pi klasa prowadząca wygra pełne
                        głosowanie (alfa = 1 - pi w artykule)
        min_votes       ile głosów (klasy prowadzącej i drugiej łącznie) trzeba zebrać
                        przed stopem; None = wartość z artykułu dla danego alfa
        fpc             poprawka na populację skończoną (artykuł: tak)
        one_sided       przedział jednostronny (artykuł: tak)

    Tablica progów zależy tylko od T i parametrów, jest liczona w fit() w czasie O(T^2)
    i działa dla dowolnej liczby klas (nie potrzeba pliku ani danych uczących).
    """

    def __init__(self, pi=0.99, min_votes=None, fpc=True, one_sided=True, name=None):
        if not 0.5 < pi <= 1:
            raise ValueError(f"Glee: pi musi być w (0.5, 1], a jest {pi}")
        if min_votes is not None and min_votes < 1:
            raise ValueError(f"Glee: min_votes musi być >= 1, a jest {min_votes}")
        self.pi, self.alfa = pi, round(1 - pi, 12)      # bez błędu zaokrągleń 1 - 0.99
        self.min_votes, self.fpc, self.one_sided = min_votes, fpc, one_sided
        super().__init__(name or f"GLEE_pi{pi_tag(pi)}")

    def fit(self, rf, X_train=None):
        super().fit(rf)
        self.prog = tablica_glee(self.T, self.alfa, self.min_votes, self.fpc, self.one_sided)
        return self

    def _stop_batch(self, votes):
        return czy_stop_glee(votes, self.prog)

    def _stop_one(self, votes):
        t = np.sort(votes)
        return t[-1] >= self.prog[t[-2]]
