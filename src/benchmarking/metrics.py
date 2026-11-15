"""Metryki benchmarku liczone z surowych wyników (opis: implementacja.md, rozdz. 9.5).

W pliku wyników są tylko dane surowe: predykcje, liczby drzew i czasy. Metryki liczy się
z nich przy analizie, więc zmiana albo nowa metryka obejmuje też stare przebiegi."""
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, root_mean_squared_error)

from methods import FullAveraging, FullVoting, SklearnRF
from methods.base import FloatArray, IntArray, Labels

REF_RF = SklearnRF.name
REF_FULL = FullVoting.name                      # w regresji: REF_FULL_REG
REF_FULL_REG = FullAveraging.name

# kolumny pliku wyników, które zmieniają się w obrębie przebiegu (date też: metoda może być
# policzona w późniejszym uruchomieniu); pozostałe (dataset, seed, ustawienia) są wspólne dla
# przebiegu i odróżniają go od innych
ARRAY_COLS = ["y_true", "y_pred", "n_used"]     # tablice: jedna wartość na próbkę
ROW_COLS = ["mode", "method", "pi", "delta", "time_s", "date"]


def method_metrics(names: Sequence[str], y_true: Labels, Y: Labels, N: IntArray,
                   times: FloatArray, T: int, task: str,
                   deltas: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Tabela metryk jednego przebiegu w jednym scenariuszu, wiersz na metodę.
    Wiersz i macierzy Y (predykcje) i N (liczby drzew) oraz times[i] to metoda names[i]."""
    deltas = deltas or {}
    names = list(names)
    is_regression = task == "regression"
    full = REF_FULL_REG if is_regression else REF_FULL
    # kolumny agree_* i speedup_vs_* są tylko dla metod odniesienia obecnych w przebiegu
    refs = {k: names.index(name) for k, name in (("rf", REF_RF), ("full", full))
            if name in names}
    n = Y.shape[1]

    rows = []
    for i, name in enumerate(names):
        y = Y[i]
        if is_regression:
            row = {
                "method": name,
                "mae": mean_absolute_error(y_true, y),
                "rmse": root_mean_squared_error(y_true, y),
                "r2": r2_score(y_true, y),
            }
            for k, j in refs.items():
                # zgodność: różnica względem odniesienia w tolerancji metody; metody
                # bez tolerancji muszą dać ten sam wynik (z dokładnością arytmetyki)
                diff = np.abs(y - Y[j])
                agree = diff <= deltas[name] if name in deltas else np.isclose(y, Y[j])
                row[f"agree_{k}"] = agree.mean()
                row[f"mad_{k}"] = diff.mean()
                row[f"maxdiff_{k}"] = diff.max()
        else:
            row = {
                "method": name,
                "accuracy": accuracy_score(y_true, y),
                "precision": precision_score(y_true, y, average="macro", zero_division=0),
                "recall": recall_score(y_true, y, average="macro", zero_division=0),
                "f1": f1_score(y_true, y, average="macro", zero_division=0),
            }
            for k, j in refs.items():
                row[f"agree_{k}"] = (y == Y[j]).mean()
        row["time_s"] = times[i]
        row["ms_per_sample"] = times[i] / n * 1e3
        for k, j in refs.items():
            row[f"speedup_vs_{k}"] = times[j] / times[i]
        row.update(
            trees_mean=N[i].mean(),
            trees_median=np.median(N[i]),
            trees_max=N[i].max(),
            trees_frac=N[i].mean() / T,
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def compute_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Metryki dla tabeli z pliku wyników: te same wiersze, tablice zamienione na metryki.

    Potrzebne kolumny: method, mode, time_s, T, task, y_true, y_pred, n_used (w regresji też
    delta). Przebiegi rozróżnia każda kolumna spoza ROW_COLS i ARRAY_COLS, więc najlepiej
    podać całą tabelę. Można ją zawęzić do części zbiorów, seedów albo scenariuszy, ale nie
    metod: bez wiersza full_voting metody tracą kolumny agree_full i speedup_vs_full."""
    missing = [c for c in ["method", "mode", "time_s", "T", "task", *ARRAY_COLS]
               if c not in raw.columns]
    if missing:
        raise ValueError(f"compute_metrics: missing columns {missing}")
    run_cols = [c for c in raw.columns if c not in ROW_COLS + ARRAY_COLS]
    parts = []
    for _, g in raw.groupby([*run_cols, "mode"], sort=False, dropna=False):
        if g["method"].duplicated().any():
            raise ValueError(f"compute_metrics: the same method twice in one run "
                             f"({g.iloc[0][run_cols].to_dict()}); the columns that tell "
                             f"runs apart are probably missing")
        deltas = (g.set_index("method")["delta"].dropna().to_dict() if "delta" in g
                  else {})
        m = method_metrics(g["method"].tolist(), g["y_true"].iloc[0], np.stack(g["y_pred"]),
                           np.stack(g["n_used"]), g["time_s"].to_numpy(dtype=float),
                           int(g["T"].iloc[0]), g["task"].iloc[0], deltas)
        parts.append(g.drop(columns=[*ARRAY_COLS, "time_s"]).join(m, on="method"))
    return pd.concat(parts).loc[raw.index]
