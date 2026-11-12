"""Benchmark metod inferencji (pakiet methods) na wytrenowanych lasach; nie zna config.toml."""
from .benchmark import benchmark
from .data import DATA_DIR, train_forest, wczytaj, znajdz_zbiory
from .forest_methods import methods_rf
from .modes import SUPPORTED_MODES, Mode, check_modes
from .suite import benchmark_suite

__all__ = ["benchmark", "benchmark_suite", "methods_rf", "SUPPORTED_MODES", "Mode", "check_modes",
           "DATA_DIR", "train_forest", "wczytaj", "znajdz_zbiory"]
