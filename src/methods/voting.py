"""Metody odniesienia: pełny las bez przyspieszania.

SklearnRF  - soft voting: rf.predict ze scikit-learn uśrednia prawdopodobieństwa drzew.
FullVoting - hard voting: każde pojedyncze drzewo oddaje jeden głos.
"""
import numpy as np

from .base import FixedVoting, InferenceMethod


class SklearnRF(InferenceMethod):
    """Oryginalny rf.predict (soft voting, uśrednia prawdopodobieństwa drzew)."""

    name = "rf_sklearn"

    def predict_batch(self, X):
        return self.rf.predict(X), np.full(len(X), self.T)

    def predict_one(self, x):
        return self.rf.predict(x)[0], self.T


class FullVoting(FixedVoting):
    """Pełne głosowanie większościowe (hard voting) wszystkich drzew
    (pętle głosowania są w FixedVoting)."""

    name = "full_voting"
