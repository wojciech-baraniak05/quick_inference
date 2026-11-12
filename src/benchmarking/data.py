"""Zbiory danych i trening lasu do benchmarku."""
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier

from methods.base import FloatArray

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


def wczytaj(plik: str | Path) -> tuple[FloatArray, NDArray[Any]]:
    """(X, y) z pliku .parquet: cechy to wszystkie kolumny poza target, kolumny
    nieliczbowe są zamieniane na kody całkowite."""
    df = pd.read_parquet(plik)
    cechy = df.drop(columns="target")
    for c in cechy.select_dtypes(exclude="number").columns:
        cechy[c] = cechy[c].astype("category").cat.codes
    return cechy.to_numpy(dtype=float), df["target"].to_numpy()


def train_forest(X_train: ArrayLike, y_train: ArrayLike, n_trees: int,
                 max_depth: int | None = None, seed: int | None = None) -> RandomForestClassifier:
    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    # trening równolegle, predykcja w jednym wątku: uczciwe porównanie czasu z metodami,
    # które odpytują drzewa po kolei
    rf.set_params(n_jobs=None)
    return rf
