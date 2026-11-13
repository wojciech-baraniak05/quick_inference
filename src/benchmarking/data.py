"""Zbiory danych i trening lasu do benchmarku."""
from collections.abc import Iterable
from pathlib import Path
from typing import Any, get_args

import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from methods.base import FloatArray, Forest, Task

TASKS: tuple[Task, ...] = get_args(Task)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def znajdz_zbiory(sciezki: Iterable[str]) -> list[Path]:
    """Pliki .parquet ze ścieżek względem data/ (folder = wszystkie pliki w nim)."""
    pliki: list[Path] = []
    for s in sciezki:
        p = DATA_DIR / s
        if p.is_dir():
            pliki += sorted(p.glob("*.parquet"))
        elif p.is_file():
            pliki.append(p)
        else:
            raise FileNotFoundError(f"brak zbioru danych {p}")
    return pliki


def wczytaj(plik: str | Path, pomin: Iterable[str] = ()) -> tuple[FloatArray, NDArray[Any]]:
    """(X, y) z pliku .parquet: cechy to wszystkie kolumny poza target i poza pomin
    (kolumny dodatkowe, np. year i CO w gas_turbine; brak takiej kolumny nie jest błędem),
    kolumny nieliczbowe są zamieniane na kody całkowite."""
    df = pd.read_parquet(plik)
    cechy = df.drop(columns="target").drop(columns=list(pomin), errors="ignore")
    for c in cechy.select_dtypes(exclude="number").columns:
        cechy[c] = cechy[c].astype("category").cat.codes
    return cechy.to_numpy(dtype=float), df["target"].to_numpy()


def train_forest(X_train: ArrayLike, y_train: ArrayLike, n_trees: int,
                 max_depth: int | None = None, seed: int | None = None,
                 task: Task = "classification") -> Forest:
    """Las klasyfikacyjny albo, dla task="regression", regresyjny."""
    if task not in TASKS:
        raise ValueError(f"nieznane zadanie {task!r}, dozwolone: {TASKS}")
    klasa = RandomForestRegressor if task == "regression" else RandomForestClassifier
    rf = klasa(n_estimators=n_trees, max_depth=max_depth, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    # trening równolegle, predykcja w jednym wątku: uczciwe porównanie czasu z metodami,
    # które odpytują drzewa po kolei
    rf.set_params(n_jobs=None)
    return rf
