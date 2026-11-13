"""Metody inferencji dla lasu losowego, każda w osobnym module (opis: implementacja.md)."""
from .base import FastRegTree, FastTree, FixedVoting, InferenceMethod, SequentialPruning
from .glee import Glee
from .hyper import Hyper
from .ib_pruning import IBPruning, czy_stop_wektorowo
from .margin import MarginEarlyStop
from .own import Own, OwnRegression
from .regression import (FullAveraging, GleeRegression, IBRegression, RegressionMethod,
                         SequentialRegression, SklearnRFReg)
from .registry import METHOD_TYPES, build_method, build_methods
from .voting import FullVoting, SklearnRF

__all__ = ["InferenceMethod", "FastTree", "FastRegTree", "FixedVoting", "SequentialPruning",
           "SklearnRF", "FullVoting", "IBPruning", "Hyper", "Glee", "MarginEarlyStop", "Own",
           "RegressionMethod", "SequentialRegression", "SklearnRFReg", "FullAveraging",
           "IBRegression", "GleeRegression", "OwnRegression", "czy_stop_wektorowo",
           "METHOD_TYPES", "build_method", "build_methods"]
