"""Wspólny interfejs metod inferencji dla lasu losowego."""
import numpy as np


class InferenceMethod:
    """Bazowa klasa metod inferencji.

    predict_batch(X) -> (y_pred, n_used)   dla macierzy X (n_próbek, n_cech)
    predict_one(x)   -> (y, n_used)        dla jednej próbki x o kształcie (1, n_cech)
    n_used = liczba drzew odpytanych dla danej próbki.

    Nową metodę dodaje się jako podklasę z predict_batch i predict_one
    (metodę z wczesnym stopem: jako podklasę SequentialPruning z samym _stop_batch,
    metodę głosującą stałym podzbiorem drzew: jako podklasę FixedVoting z samym voters).
    Żeby dało się ją wybrać w config.toml, trzeba ją jeszcze dopisać
    do METHOD_TYPES w methods/registry.py.
    """

    name = "method"

    def __init__(self, name=None):
        if name is not None:
            self.name = name

    def fit(self, rf, X_train=None, y_train=None):
        """Wiąże metodę z lasem rf. X_train, y_train (dane, na których uczono las) są
        potrzebne tylko metodom, które coś z nich szacują (Hyper: X_train,
        MarginEarlyStop: X_train i y_train)."""
        self.rf = rf
        self.trees = rf.estimators_
        self.classes = rf.classes_
        self.T = len(self.trees)
        self.l = len(self.classes)
        return self

    def predict_batch(self, X):
        raise NotImplementedError

    def predict_one(self, x):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"


class FixedVoting(InferenceMethod):
    """Hard voting stałej listy drzew self.voters, tej samej dla każdej próbki.

    Po fit() głosują wszystkie drzewa lasu (FullVoting); podklasa może w swoim fit()
    wybrać podzbiór i kolejność (np. MarginEarlyStop: N* pierwszych drzew rankingu).
    """

    def fit(self, rf, X_train=None, y_train=None):
        super().fit(rf)
        self.voters = self.trees
        return self

    def predict_batch(self, X):
        votes = np.zeros((len(X), self.l), dtype=int)
        wiersze = np.arange(len(X))
        for tree in self.voters:
            votes[wiersze, tree.predict(X).astype(int)] += 1
        return self.classes[votes.argmax(axis=1)], np.full(len(X), len(self.voters))

    def predict_one(self, x):
        votes = np.zeros(self.l, dtype=int)
        for tree in self.voters:
            votes[int(tree.predict(x)[0])] += 1
        return self.classes[votes.argmax()], len(self.voters)


class SequentialPruning(InferenceMethod):
    """Metody z wczesnym stopem: drzewa są pytane po kolei (hard voting),
    a głosowanie próbki kończy się, gdy reguła stopu uzna wynik za pewny.

    Podklasa podaje tylko regułę stopu:
        _stop_batch(votes) -> maska bool (n,)   dla votes o kształcie (n, l)
        _stop_one(votes)   -> bool              dla votes o kształcie (l,)
                                                (opcjonalnie, szybsza wersja skalarna)
    """

    def _stop_batch(self, votes):
        raise NotImplementedError

    def _stop_one(self, votes):
        return bool(self._stop_batch(votes[None, :])[0])

    def predict_batch(self, X):
        votes = np.zeros((len(X), self.l), dtype=int)
        n_used = np.full(len(X), self.T)
        aktywne = np.arange(len(X))                 # próbki jeszcze niezdecydowane
        for k, tree in enumerate(self.trees):
            votes[aktywne, tree.predict(X[aktywne]).astype(int)] += 1
            stop = self._stop_batch(votes[aktywne])
            n_used[aktywne[stop]] = k + 1
            aktywne = aktywne[~stop]
            if len(aktywne) == 0:
                break
        return self.classes[votes.argmax(axis=1)], n_used

    def predict_one(self, x):
        votes = np.zeros(self.l, dtype=int)
        for k, tree in enumerate(self.trees):
            votes[int(tree.predict(x)[0])] += 1
            if self._stop_one(votes):
                break
        return self.classes[votes.argmax()], k + 1
