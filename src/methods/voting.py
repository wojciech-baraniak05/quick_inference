"""Metody odniesienia: pełny las bez przyspieszania."""
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .base import FixedVoting, InferenceMethod, Prediction


class SklearnRF(InferenceMethod):
    """Oryginalny rf.predict: soft voting, czyli średnia prawdopodobieństw drzew."""

    name = "rf_sklearn"

    def predict_batch(self, X: ArrayLike) -> Prediction:
        y = self.rf.predict(X)
        return y, np.full(len(y), self.T)

    def predict_one(self, x: ArrayLike) -> tuple[Any, int]:
        return self.rf.predict(x)[0], self.T


class FullVoting(FixedVoting):
    """Hard voting wszystkich drzew: każde drzewo oddaje jeden głos."""

    name = "full_voting"
