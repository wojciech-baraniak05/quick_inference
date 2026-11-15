"""Podsumowanie pliku wyników: średnie po seedach i sprawdzenie agree_full >= pi.

Uruchomienie z src/:  python -m analysis.summary [plik.parquet]
(domyślnie out_file z config.toml). Opis: implementacja.md, rozdz. 9.6."""
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from benchmarking import run_columns

from .metrics import compute_metrics

# kolumny jakości zależą od zadania; summary() pokazuje te, które są w wynikach
SUMMARY_COLS = ["accuracy", "f1", "mae", "rmse", "r2", "agree_rf", "agree_full", "mad_full",
                "ms_per_sample", "speedup_vs_rf", "speedup_vs_full", "trees_mean", "trees_frac"]


def group_columns(raw: pd.DataFrame) -> list[str]:
    """dataset i ustawienia, które zmieniają się w obrębie jakiegoś zbioru (np. dwa max_depth
    w pliku). Przebiegi o różnych ustawieniach nie są więc uśredniane razem; seed jest
    powtórzeniem, po którym się uśrednia."""
    varying = [c for c in run_columns(raw) if c not in ("dataset", "seed")
               and raw.groupby("dataset")[c].nunique(dropna=False).max() > 1]
    return ["dataset", *varying]


def aggregate(m: pd.DataFrame, by: Sequence[str] = ("dataset",), mode: str = "batch",
              stats: str | Sequence[str] = ("mean", "std")) -> pd.DataFrame:
    """Statystyki metryk po seedach, wiersz na (by..., method); by=() uśrednia po wszystkim."""
    df = m[m["mode"] == mode]
    metrics = [c for c in SUMMARY_COLS if c in df.columns]
    stats = stats if isinstance(stats, str) else list(stats)
    return df.groupby([*by, "method"], sort=False, dropna=False)[metrics].agg(stats)


def summary(m: pd.DataFrame, by: Sequence[str] = ("dataset",)) -> None:
    with pd.option_context("display.width", 250, "display.max_columns", None,
                           "display.max_rows", None, "display.float_format", "{:.4f}".format):
        for mode in m["mode"].unique():
            n_seeds = m.loc[m["mode"] == mode, "seed"].nunique()
            print(f"\n[{mode}] mean over {n_seeds} seeds, per {', '.join(by)}:")
            print(aggregate(m, by, mode, stats="mean"))
            print(f"\n[{mode}] mean over all runs:")
            print(aggregate(m, (), mode, stats="mean"))


def check_pi(m: pd.DataFrame, by: Sequence[str] = ("dataset",)) -> None:
    """Przebiegi, w których zgodność metody z pełnym lasem spadła poniżej jej pi
    (klasyfikacja: ta sama klasa, regresja: różnica w tolerancji delta metody)."""
    if "agree_full" not in m:
        print("agree_full check skipped: no full_voting / full_averaging in the results")
        return
    checked = m[m["pi"].notna() & m["agree_full"].notna()]
    for mode in checked["mode"].unique():
        rows = checked[checked["mode"] == mode]
        counts, failed = [], []
        for name, g in rows.groupby("method", sort=False):
            below = g[g["agree_full"] < g["pi"]]
            counts.append(f"{name} {len(below)}/{len(g)}")
            if len(below):
                failed.append((name, below))
        print(f"[{mode}] runs with agree_full < pi: {', '.join(counts)}")
        for name, below in failed:
            cols = [*by, "seed", "pi", "agree_full"]
            print(f"  {name}:\n{below[cols].to_string(index=False)}")


def main(path: str | Path) -> None:
    raw = pd.read_parquet(path)
    m = compute_metrics(raw)
    by = group_columns(raw)
    print(f"{path}: {len(raw)} rows, {raw.groupby(run_columns(raw), dropna=False).ngroups} runs")
    summary(m, by)
    print()
    check_pi(m, by)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        import config as cfg
        from benchmarking import DATA_DIR
        main(DATA_DIR / cfg.OUT_FILE)
