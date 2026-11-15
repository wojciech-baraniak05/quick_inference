"""Benchmark metod inferencji (pakiet methods) na wytrenowanych lasach; nie zna config.toml.
Zapisuje surowe wyniki do pliku Parquet; metryki i analiza są w src/analysis."""
from .benchmark import Benchmark
from .data import DATA_DIR, find_datasets, load_dataset, train_forest
from .forest_methods import ForestMethods
from .modes import SUPPORTED_MODES, Mode, check_modes
from .storage import ARRAY_COLS, ROW_COLS, run_columns, store_runs
from .suite import BenchmarkSuite

__all__ = ["Benchmark", "BenchmarkSuite", "ForestMethods", "SUPPORTED_MODES", "Mode",
           "check_modes", "DATA_DIR", "find_datasets", "load_dataset", "train_forest",
           "ARRAY_COLS", "ROW_COLS", "run_columns", "store_runs"]
