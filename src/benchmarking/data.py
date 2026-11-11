"""Zbiory danych i trening lasu do benchmarku."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def znajdz_zbiory(sciezki):
    """Pliki .parquet ze ścieżek względem data/ (folder = wszystkie pliki w nim)."""
    pliki = []
    for s in sciezki:
        p = DATA_DIR / s
        if p.is_dir():
            pliki += sorted(p.glob("*.parquet"))
        elif p.is_file():
            pliki.append(p)
        else:
            raise FileNotFoundError(f"brak zbioru danych {p}")
    return pliki


def wczytaj(plik):
    """Zwraca (X, y); kolumny kategoryczne/tekstowe zamieniane na kody całkowite."""
    df = pd.read_parquet(plik)
    cechy = df.drop(columns="target")
    for c in cechy.select_dtypes(exclude="number").columns:
        cechy[c] = cechy[c].astype("category").cat.codes
    return cechy.to_numpy(dtype=float), df["target"].to_numpy()


def train_forest(X_train, y_train, n_trees, max_depth=None, seed=None):
    """Trening równolegle, potem n_jobs=None, żeby predykcja szła w jednym wątku
    (uczciwe porównanie czasów z metodami odpytującymi drzewa po kolei)."""
    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf.set_params(n_jobs=None)
    return rf
