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

    order - kolejność odpytywania drzew:
        "fixed"  - zawsze kolejność rf.estimators_,
        "random" - każda próbka dostaje własną losową permutację drzew
                   (seed ustala generator; jest tworzony od nowa w fit()).
    """

    ORDERS = ("fixed", "random")

    def __init__(self, name=None, order="fixed", seed=None):
        if order not in self.ORDERS:
            raise ValueError(f"order musi być jednym z {self.ORDERS}, a jest {order!r}")
        self.order, self.seed = order, seed
        super().__init__(name)

    def fit(self, rf, X_train=None):
        super().fit(rf, X_train)
        self.rng = np.random.default_rng(self.seed)
        return self

    def _stop_batch(self, votes):
        raise NotImplementedError

    def _stop_one(self, votes):
        return bool(self._stop_batch(votes[None, :])[0])

    def predict_batch(self, X):
        if self.order == "random":
            return self._predict_batch_random(X)
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

    def _predict_batch_random(self, X):
        """predict_batch z osobną losową permutacją drzew dla każdej próbki.
        W kroku k próbka i pyta drzewo perm[i, k]; próbki pytające to samo drzewo
        są grupowane w jedno wywołanie predict."""
        n = len(X)
        votes = np.zeros((n, self.l), dtype=int)
        n_used = np.full(n, self.T)
        perm = np.tile(np.arange(self.T, dtype=np.min_scalar_type(self.T)), (n, 1))
        perm = self.rng.permuted(perm, axis=1)
        aktywne = np.arange(n)
        for k in range(self.T):
            drzewa = perm[aktywne, k]
            idx = np.argsort(drzewa, kind="stable")
            probki, drzewa = aktywne[idx], drzewa[idx]
            granice = np.flatnonzero(np.diff(drzewa)) + 1
            for grupa, j in zip(np.split(probki, granice), drzewa[np.r_[0, granice]]):
                votes[grupa, self.trees[j].predict(X[grupa]).astype(int)] += 1
            stop = self._stop_batch(votes[aktywne])
            n_used[aktywne[stop]] = k + 1
            aktywne = aktywne[~stop]
            if len(aktywne) == 0:
                break
        return self.classes[votes.argmax(axis=1)], n_used

    def predict_one(self, x):
        votes = np.zeros(self.l, dtype=int)
        trees = self.trees
        if self.order == "random":
            trees = [self.trees[j] for j in self.rng.permutation(self.T)]
        for k, tree in enumerate(trees):
            votes[int(tree.predict(x)[0])] += 1
            if self._stop_one(votes):
                break
        return self.classes[votes.argmax()], k + 1
