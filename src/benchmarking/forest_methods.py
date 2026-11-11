"""Zbiór metod inferencji dopasowanych do jednego lasu."""
import time

import numpy as np

from methods import FullVoting, SklearnRF

from .modes import check_modes


class methods_rf:
    """Zbiór metod inferencji dla jednego wytrenowanego lasu losowego.

    Po run(X, mode) wyniki są w macierzach:
        Y_pred[mode]  (n_metod, n_próbek)  predykcje
        N_used[mode]  (n_metod, n_próbek)  liczba odpytanych drzew
        times[mode]   (n_metod,)           czas predykcji w s (mediana z n_repeats)
    Wiersz i odpowiada metodzie names[i].

    X_train (dane uczące lasu) jest przekazywane do fit() każdej metody;
    potrzebują go metody szacujące coś z danych (np. Hyper).
    """

    def __init__(self, rf, methods=None, X_train=None):
        self.rf = rf
        self.X_train = X_train
        self.classes = rf.classes_
        self.T = len(rf.estimators_)
        self.l = len(rf.classes_)
        self.methods = []
        for m in (methods if methods is not None else [SklearnRF(), FullVoting()]):
            self.add_method(m)
        self.Y_pred, self.N_used, self.times = {}, {}, {}

    @property
    def names(self):
        return [m.name for m in self.methods]

    def add_method(self, method):
        if method.name in self.names:
            raise ValueError(f"metoda o nazwie '{method.name}' już istnieje")
        self.methods.append(method.fit(self.rf, self.X_train))
        return self

    def index(self, name):
        return self.names.index(name)

    def _run_method(self, method, X, mode):
        if mode == "batch":
            return method.predict_batch(X)
        wyniki = [method.predict_one(x[None, :]) for x in X]
        y, n = zip(*wyniki)
        return np.array(y), np.array(n)

    def run(self, X, mode="batch", n_repeats=1):
        check_modes([mode])
        Y = np.empty((len(self.methods), len(X)), dtype=self.classes.dtype)
        N = np.empty((len(self.methods), len(X)), dtype=int)
        times = np.empty(len(self.methods))
        for i, method in enumerate(self.methods):
            pomiary = []
            for _ in range(n_repeats):
                start = time.perf_counter()
                y, n = self._run_method(method, X, mode)
                pomiary.append(time.perf_counter() - start)
            Y[i], N[i], times[i] = y, n, np.median(pomiary)
        self.Y_pred[mode], self.N_used[mode], self.times[mode] = Y, N, times
        return Y, N, times
