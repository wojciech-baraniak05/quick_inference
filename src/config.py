"""Project configuration read from config.toml in the repository root."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.toml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"{CONFIG_PATH} not found - copy config.example.toml to config.toml and adjust.")

with CONFIG_PATH.open("rb") as f:
    _config = tomllib.load(f)

SEED: int = int(_config["seed"])
