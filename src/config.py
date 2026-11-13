"""Project configuration read from config.toml in the repository root."""
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
OUT_DIR: str = _bench["out_dir"]
MODES: tuple[str, ...] = tuple(_bench["modes"])
SEEDS: list[int] = [int(s) for s in _bench["seeds"]]
TEST_SIZE: float = float(_bench["test_size"])
N_ONLINE: int = int(_bench["n_online"])
N_REPEATS: int = int(_bench["n_repeats"])
SKLEARN_SOFT_VOTING: bool = bool(_bench["sklearn_soft_voting"])

# [forest]
_forest = _config["forest"]
N_TREES: int = int(_forest["n_trees"])
MAX_DEPTH: int | None = _forest.get("max_depth")   # brak klucza = bez limitu głębokości

# [[methods]]: jeden słownik na blok, np. {"type": "ib_pruning", "pi": 0.98}
METHODS: list[dict[str, Any]] = list(_config.get("methods", []))
