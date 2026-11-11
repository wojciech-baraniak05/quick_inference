"""Benchmark metod inferencji (pakiet methods) na wytrenowanych lasach.

modes           - dostępne scenariusze (batch, online)
forest_methods  - methods_rf: metody dopasowane do jednego lasu
benchmark       - benchmark: jeden las, jeden zbiór testowy
suite           - benchmark_suite: wiele zbiorów i powtórzeń
data            - wczytywanie zbiorów i trening lasu
"""
from .benchmark import benchmark
from .data import DATA_DIR, train_forest, wczytaj, znajdz_zbiory
from .forest_methods import methods_rf
from .modes import SUPPORTED_MODES, check_modes
from .suite import benchmark_suite

__all__ = ["benchmark", "benchmark_suite", "methods_rf", "SUPPORTED_MODES", "check_modes",
           "DATA_DIR", "train_forest", "wczytaj", "znajdz_zbiory"]
