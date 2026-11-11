"""Metody odniesienia: pełny las bez przyspieszania.

SklearnRF  - soft voting: rf.predict ze scikit-learn uśrednia prawdopodobieństwa drzew.
FullVoting - hard voting: każde pojedyncze drzewo oddaje jeden głos.
"""
import numpy as np

from .base import InferenceMethod


class SklearnRF(InferenceMethod):
    """Oryginalny rf.predict (soft voting, uśrednia prawdopodobieństwa drzew)."""

    name = "rf_sklearn"

    def predict_batch(self, X):
        return self.rf.predict(X), np.full(len(X), self.T)

    def predict_one(self, x):
        return self.rf.predict(x)[0], self.T


class FullVoting(InferenceMethod):
    """Pełne głosowanie większościowe (hard voting) wszystkich drzew."""

    name = "full_voting"

    def predict_batch(self, X):
        votes = np.zeros((len(X), self.l), dtype=int)
        wiersze = np.arange(len(X))
        for tree in self.trees:
            votes[wiersze, tree.predict(X).astype(int)] += 1
        return self.classes[votes.argmax(axis=1)], np.full(len(X), self.T)

    def predict_one(self, x):
        votes = np.zeros(self.l, dtype=int)
        for tree in self.trees:
            votes[int(tree.predict(x)[0])] += 1
        return self.classes[votes.argmax()], self.T
