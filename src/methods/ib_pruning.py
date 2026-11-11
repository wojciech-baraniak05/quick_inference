"""Statystyczny instance-based pruning (Hernández-Lobato i in., 2009)."""
from functools import lru_cache

import numpy as np

from generate_table_IB import TABLE_DIR, czy_stop, pi_tag, plik_tablicy

from .base import SequentialPruning


@lru_cache(maxsize=None)
def _wczytaj_npz(path):
    d = np.load(path)
    return d["prog"], float(d["pi"])


def wczytaj_tablice(plik, podpowiedz=""):
    """(prog, pi) z pliku .npz; ścieżka względna jest liczona od data/tablice/."""
    path = TABLE_DIR / plik                         # ścieżka bezwzględna zostaje bez zmian
    if not path.exists():
        raise FileNotFoundError(f"brak tablicy progów {path}{podpowiedz}")
    return _wczytaj_npz(str(path))


def czy_stop_wektorowo(votes, prog):
    """czy_stop dla wielu próbek naraz; votes ma kształt (n_próbek, l)."""
    t = -np.sort(-votes, axis=1)                    # głosy malejąco w każdym wierszu
    return t[:, 0] >= prog[tuple(t[:, 1:].T)]


class IBPruning(SequentialPruning):
    """Drzewa są pytane po kolei, a głosowanie kończy się, gdy wynik jest pewny
    z prawdopodobieństwem >= pi (według tablicy progów prog).

    Tablicę można podać na trzy sposoby:
        IBPruning(pi=0.98)                    - plik T{T}_L{l}_pi98.npz dobierany
                                                w fit() do rozmiaru lasu,
        IBPruning(table="T501_L2_pi98.npz")   - konkretny plik,
        IBPruning(prog, pi=0.98)              - gotowa tablica.
    """

    def __init__(self, prog=None, pi=None, table=None, name=None):
        if table is not None:
            prog, pi = wczytaj_tablice(table)
        if prog is None and pi is None:
            raise ValueError("IBPruning: podaj pi, table albo prog")
        self._prog = None if prog is None else np.asarray(prog)
        self.pi = pi
        if name is None:
            name = f"IB_pi{pi_tag(pi)}" if pi is not None else "ib_pruning"
        super().__init__(name)

    def fit(self, rf, X_train=None):
        super().fit(rf)
        if self._prog is not None:
            self.prog = self._prog
        else:
            self.prog = wczytaj_tablice(
                plik_tablicy(self.T, self.l, self.pi),
                f"; wygeneruj ją w src/: from generate_table_IB import generuj_tablice; "
                f"generuj_tablice(T={self.T}, l={self.l}, pi={self.pi})")[0]
        T_prog, l_prog = self.prog.shape[0] - 1, self.prog.ndim + 1
        if (T_prog, l_prog) != (self.T, self.l):
            raise ValueError(f"{self.name}: tablica jest dla T={T_prog}, l={l_prog}, "
                             f"a las ma T={self.T}, l={self.l}")
        return self

    def _stop_batch(self, votes):
        return czy_stop_wektorowo(votes, self.prog)

    def _stop_one(self, votes):
        return czy_stop(votes, self.prog)
