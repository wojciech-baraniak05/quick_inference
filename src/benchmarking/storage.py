"""Plik wyników: wszystkie przebiegi w jednym pliku Parquet (opis: implementacja.md, rozdz. 9.5).

Wiersz na (przebieg, scenariusz, metoda); tylko dane surowe. Metryki liczy src/analysis."""
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# kolumny, które zmieniają się w obrębie przebiegu (date też: metoda może być policzona
# w późniejszym uruchomieniu); pozostałe (dataset, seed, ustawienia) są wspólne dla
# przebiegu i odróżniają go od innych
ARRAY_COLS = ["y_true", "y_pred", "n_used"]     # tablice: jedna wartość na próbkę
ROW_COLS = ["mode", "method", "pi", "delta", "time_s", "date"]


def run_columns(df: pd.DataFrame) -> list[str]:
    """Kolumny przebiegu (dataset, seed, ustawienia): wszystkie spoza ROW_COLS i ARRAY_COLS."""
    return [c for c in df.columns if c not in ROW_COLS + ARRAY_COLS]


def _row_ids(df: pd.DataFrame, cols: list[str]) -> list[tuple[object, ...]]:
    """Tożsamość każdego wiersza: krotka wartości cols, brak wartości jako None
    (np. kolumna, której starsze przebiegi nie miały)."""
    vals = df.reindex(columns=cols).astype(object)
    vals = vals.where(vals.notna(), None)
    return list(map(tuple, vals.to_numpy()))


def store_runs(path: str | Path, frames: Iterable[pd.DataFrame]) -> Path:
    """Dopisuje przebiegi (Benchmark.to_frame()) do pliku Parquet. Wiersz zastępuje zapisany
    wiersz o tych samych ustawieniach przebiegu (dataset, seed, n_trees...; bez date),
    scenariuszu i metodzie. Pozostałe wiersze zostają, także metody, których nowy przebieg
    nie liczył. Zapis przez plik tymczasowy i podmianę."""
    path = Path(path)
    table = pd.read_parquet(path) if path.exists() else None
    for new in frames:                              # po kolei: późniejszy wygrywa
        if table is not None:
            cols = [*sorted({*run_columns(table), *run_columns(new)}), "mode", "method"]
            replaced = set(_row_ids(new, cols))
            table = table[[row not in replaced for row in _row_ids(table, cols)]]
        table = new if table is None else pd.concat([table, new], ignore_index=True)
    if table is None:
        raise ValueError("store_runs: no runs to store")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    table.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    return path
