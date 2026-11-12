"""Metody inferencji dla lasu losowego: każda w osobnym module.

base        - InferenceMethod (wspólny interfejs), FixedVoting (stała lista drzew),
              SequentialPruning (wczesny stop zależny od próbki)
voting      - SklearnRF (soft voting), FullVoting (hard voting)
ib_pruning  - IBPruning (statystyczny instance-based pruning, tablica z pliku)
hyper       - Hyper (IB pruning z a priori rozkładu głosów szacowanym z danych)
glee        - Glee (wczesny stop z gaussowskim przedziałem ufności, artykuł COMET)
margin      - MarginEarlyStop (ranking drzew + liczba drzew z marginesu lasu)
oob         - predykcje out-of-bag na danych uczących (dla hyper i margin)
registry    - budowanie metod z config.toml
"""
from .base import FixedVoting, InferenceMethod, SequentialPruning
from .glee import Glee
from .hyper import Hyper
from .ib_pruning import IBPruning, czy_stop_wektorowo
from .margin import MarginEarlyStop
from .registry import METHOD_TYPES, build_method, build_methods
from .voting import FullVoting, SklearnRF

__all__ = ["InferenceMethod", "FixedVoting", "SequentialPruning", "SklearnRF", "FullVoting",
           "IBPruning", "Hyper", "Glee", "MarginEarlyStop", "czy_stop_wektorowo",
           "METHOD_TYPES", "build_method", "build_methods"]
