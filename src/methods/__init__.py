"""Metody inferencji dla lasu losowego: każda w osobnym module.

base        - InferenceMethod (wspólny interfejs), SequentialPruning (wczesny stop)
voting      - SklearnRF (soft voting), FullVoting (hard voting)
ib_pruning  - IBPruning (statystyczny instance-based pruning, tablica z pliku)
hyper       - Hyper (IB pruning z a priori rozkładu głosów szacowanym z danych)
glee        - Glee (wczesny stop z gaussowskim przedziałem ufności, artykuł COMET)
registry    - budowanie metod z config.toml
"""
from .base import InferenceMethod, SequentialPruning
from .glee import Glee
from .hyper import Hyper
from .ib_pruning import IBPruning, czy_stop_wektorowo
from .registry import METHOD_TYPES, build_method, build_methods
from .voting import FullVoting, SklearnRF

__all__ = ["InferenceMethod", "SequentialPruning", "SklearnRF", "FullVoting", "IBPruning",
           "Hyper", "Glee", "czy_stop_wektorowo",
           "METHOD_TYPES", "build_method", "build_methods"]
