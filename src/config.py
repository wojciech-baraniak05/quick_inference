"""Konfiguracja projektu wczytana z config.toml w katalogu głównym repozytorium."""
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.toml"

with CONFIG_PATH.open("rb") as f:
    _config: dict[str, Any] = tomllib.load(f)

SEED: int = int(_config["seed"])

# [benchmark]
_bench = _config["benchmark"]
TASK: str = _bench.get("task", "classification")    # "classification" | "regression"
DATASETS: list[str] = list(_bench["datasets"])
DROP_COLUMNS: list[str] = list(_bench.get("drop_columns", []))  # kolumny, które nie są cechami
if "out_dir" in _bench:
    raise ValueError('[benchmark] out_dir was replaced by out_file (one Parquet file for all '
                     'runs), e.g. out_file = "results/binary.parquet"')
OUT_FILE: str = _bench["out_file"]                  # plik wyników .parquet, względem data/
MODES: tuple[str, ...] = tuple(_bench["modes"])
SEEDS: list[int] = [int(s) for s in _bench["seeds"]]
TEST_SIZE: float = float(_bench["test_size"])
# część całego zbioru odkładana z danych uczących na zbiór walidacyjny (metody z calib = "val");
# 0 = bez zbioru walidacyjnego
VAL_SIZE: float = float(_bench.get("val_size", 0.0))
N_ONLINE: int = int(_bench["n_online"])
N_REPEATS: int = int(_bench["n_repeats"])
SKLEARN_SOFT_VOTING: bool = bool(_bench["sklearn_soft_voting"])

# [forest]
FOREST_KEYS = ("n_trees", "max_depth")      # parametry lasu, które można nadpisać per zbiór


def _n_trees(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where}: n_trees must be a positive integer, got {value!r}")
    return value


def _max_depth(value: Any, where: str) -> int | None:
    """Dodatnia liczba całkowita albo "none" (bez limitu głębokości; TOML nie ma null)."""
    if isinstance(value, str) and value.lower() == "none":
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{where}: max_depth must be a positive integer or "none", '
                         f"got {value!r}")
    return value


_forest = _config["forest"]
N_TREES: int = _n_trees(_forest["n_trees"], "[forest]")
# brak klucza albo "none" = bez limitu głębokości
MAX_DEPTH: int | None = _max_depth(_forest.get("max_depth", "none"), "[forest]")

# [forest.per_dataset.<zbiór>]: nadpisania n_trees / max_depth dla zbioru o tej nazwie pliku
# (bez .parquet); brak klucza = wartość z [forest]
FOREST_PER_DATASET: dict[str, dict[str, Any]] = {}
for _name, _table in _forest.get("per_dataset", {}).items():
    _where = f"[forest.per_dataset.{_name}]"
    if not isinstance(_table, dict):
        raise ValueError(f"{_where}: expected a table of {FOREST_KEYS}, got {_table!r}")
    _unknown = sorted(set(_table) - set(FOREST_KEYS))
    if _unknown:
        raise ValueError(f"{_where}: unknown keys {_unknown}, expected from {FOREST_KEYS}")
    _params: dict[str, Any] = {}
    if "n_trees" in _table:
        _params["n_trees"] = _n_trees(_table["n_trees"], _where)
    if "max_depth" in _table:
        _params["max_depth"] = _max_depth(_table["max_depth"], _where)
    FOREST_PER_DATASET[_name] = _params


def forest_params(dataset: str) -> dict[str, Any]:
    """Parametry lasu dla zbioru: {"n_trees", "max_depth"} z [forest] i jego nadpisań."""
    return {"n_trees": N_TREES, "max_depth": MAX_DEPTH, **FOREST_PER_DATASET.get(dataset, {})}

# [[methods]]: jeden słownik na blok, np. {"type": "ib_pruning", "pi": 0.98}
METHODS: list[dict[str, Any]] = list(_config.get("methods", []))
