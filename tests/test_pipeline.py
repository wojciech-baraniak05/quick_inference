"""Wszystko wokół metod: config.py, rejestr metod, Benchmark, plik wyników (storage.py,
suite.py) i metryki (src/analysis)."""
import importlib
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import compute_metrics, method_metrics
from benchmarking import Benchmark, BenchmarkSuite, ForestMethods, store_runs
from benchmarking import suite as suite_module
from methods import (METHOD_TYPES, FullAveraging, FullVoting, Glee, IBPruning, IBRegression,
                     SklearnRF, SklearnRFReg, build_method, build_methods)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------------------

BASE_CONFIG = """
seed = 1
[benchmark]
datasets = ["generated/binary"]
out_file = "results/x.parquet"
modes = ["batch"]
seeds = [1, 2]
test_size = 0.3
n_online = 10
n_repeats = 1
sklearn_soft_voting = true
"""


@pytest.fixture
def load_config(monkeypatch, tmp_path):
    """Świeży import config z QI_CONFIG wskazującym plik z podanym tekstem; po teście
    moduł config wraca do stanu sprzed."""
    saved = sys.modules.get("config")

    def load(text: str):
        path = tmp_path / "cfg.toml"
        path.write_text(BASE_CONFIG + text, encoding="utf-8")
        monkeypatch.setenv("QI_CONFIG", str(path))
        sys.modules.pop("config", None)
        return importlib.import_module("config")

    yield load
    if saved is None:
        sys.modules.pop("config", None)
    else:
        sys.modules["config"] = saved


def test_config_defaults_and_per_dataset_override(load_config):
    cfg = load_config("""
[forest]
n_trees = 51
[forest.per_dataset.big]
max_depth = 7
[forest.per_dataset.small]
n_trees = 11
""")
    assert cfg.TASK == "classification"
    assert cfg.VAL_SIZE == 0.0 and cfg.DROP_COLUMNS == [] and cfg.METHODS == []
    assert cfg.forest_params("other") == {"n_trees": 51, "max_depth": None}  # brak = bez limitu
    assert cfg.forest_params("big") == {"n_trees": 51, "max_depth": 7}
    assert cfg.forest_params("small") == {"n_trees": 11, "max_depth": None}


@pytest.mark.parametrize("forest, match", [
    ("[forest]\nn_trees = true\n", "n_trees must be a positive integer"),
    ("[forest]\nn_trees = 51\nmax_depth = 0\n", "max_depth"),
    ("[forest]\nn_trees = 51\n[forest.per_dataset.x]\nn_tree = 3\n", "unknown keys"),
])
def test_config_forest_validation(load_config, forest, match):
    with pytest.raises(ValueError, match=match):
        load_config(forest)


# ---------------------------------------------------------------------------------------
# registry.py
# ---------------------------------------------------------------------------------------

def test_build_methods_selects_task_and_rejects_duplicates():
    specs = [{"type": "full_voting"},
             {"type": "ib_pruning", "pi": 0.95},
             {"type": "ib_pruning", "pi": 0.9, "enabled": False},
             {"type": "full_averaging"},
             {"type": "ib_reg", "pi": 0.95, "delta": 1.0}]
    assert [type(m) for m in build_methods(specs, True, "classification")] == [
        SklearnRF, FullVoting, IBPruning]
    assert [type(m) for m in build_methods(specs, True, "regression")][:2] == [
        SklearnRFReg, FullAveraging]
    with pytest.raises(ValueError, match="duplicate method names"):
        build_methods([{"type": "glee", "pi": 0.98}, {"type": "glee", "pi": 0.98}])


@pytest.mark.parametrize("path", [ROOT / "config.toml", *sorted((ROOT / "configs").glob("*.toml"))],
                         ids=lambda p: p.name)
def test_repository_configs_build(path):
    """Każdy blok [[methods]] plików z repozytorium (także enabled = false) daje się zbudować:
    literówka w nazwie parametru albo zmiana konstruktora wyszłaby dopiero w środku długiego
    benchmarku."""
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    specs = cfg.get("methods", [])
    for s in specs:
        build_method(s)
    for task in {METHOD_TYPES[s["type"]].task for s in specs}:
        assert build_methods(specs, cfg["benchmark"]["sklearn_soft_voting"], task)


# ---------------------------------------------------------------------------------------
# Benchmark i plik wyników (benchmark.py, storage.py, suite.py)
# ---------------------------------------------------------------------------------------

def test_benchmark_to_frame_layout_and_release(binary):
    m = ForestMethods(binary.rf, X_train=binary.X_train, y_train=binary.y_train)
    b = Benchmark(m, binary.X_test, binary.y_test, metadata={"dataset": "d", "seed": 1},
                  n_online=10)
    m.add_method(IBPruning(pi=0.95))                # metoda dodana po utworzeniu benchmarku
    b.run(["batch", "online"]).release()
    with pytest.raises(RuntimeError, match="release"):
        b.run(["batch"])
    df = b.to_frame()                               # wyniki zostają po release()
    assert len(df) == 2 * 3
    assert list(df["method"].unique()) == ["rf_sklearn", "full_voting", "IB_pi95"]
    batch = df[df["mode"] == "batch"].iloc[0]
    online = df[df["mode"] == "online"].iloc[0]
    assert len(batch["y_true"]) == len(binary.y_test)
    assert len(online["y_true"]) == len(online["y_pred"]) == len(online["n_used"]) == 10
    assert df.loc[df["method"] == "IB_pi95", "pi"].iloc[0] == 0.95
    assert df.loc[df["method"] == "full_voting", "pi"].isna().all()
    assert "methods" not in df.columns              # listy metadanych nie trafiają do pliku
    assert {"dataset", "seed", "T", "l", "task", "n_test", "date"} <= set(df.columns)


def frame(seed=1, methods=("full_voting", "IB_pi98"), mode="batch", date="2026-01-01",
          time_s=1.0, **settings):
    """Ramka w układzie Benchmark.to_frame()."""
    return pd.DataFrame([
        {"dataset": "d", "seed": seed, "T": 51, "task": "classification", **settings,
         "date": date, "mode": mode, "method": m, "pi": None, "delta": None, "time_s": time_s,
         "y_true": np.array([0, 1]), "y_pred": np.array([0, 1]), "n_used": np.array([51, 51])}
        for m in methods])


def test_store_runs_replaces_only_computed_methods_and_appends_other_settings(tmp_path):
    path = tmp_path / "r.parquet"
    store_runs(path, [frame(time_s=1.0, max_depth=5)])
    # to samo ustawienie, inna data, tylko jedna metoda: zastępuje tylko ją
    store_runs(path, [frame(methods=("IB_pi98",), date="2026-02-02", time_s=2.0,
                            max_depth=5)])
    by_method = pd.read_parquet(path).set_index("method")
    assert by_method.loc["IB_pi98", "time_s"] == 2.0
    assert by_method.loc["full_voting", "time_s"] == 1.0
    # inne ustawienia, seed albo tryb: dopisane obok
    store_runs(path, [frame(max_depth=None), frame(seed=2, max_depth=5),
                      frame(mode="online", max_depth=5)])
    assert len(pd.read_parquet(path)) == 8
    assert [p.name for p in tmp_path.iterdir()] == ["r.parquet"]   # bez pliku .tmp


def test_store_runs_missing_column_in_old_rows_counts_as_none(tmp_path):
    path = tmp_path / "r.parquet"
    store_runs(path, [frame()])                     # starszy przebieg bez val_size
    store_runs(path, [frame(val_size=None, time_s=3.0)])
    table = pd.read_parquet(path)
    assert len(table) == 2 and (table["time_s"] == 3.0).all()
    store_runs(path, [frame(val_size=0.1)])         # inna wartość: dopisany obok
    assert len(pd.read_parquet(path)) == 4


def test_suite_retries_failed_write(tmp_path, monkeypatch, capsys):
    class FakeBenchmark:
        def __init__(self, seed):
            self.seed = seed

        def to_frame(self):
            return frame(seed=self.seed)

    calls = {"n": 0}
    real = suite_module.store_runs

    def flaky(path, frames):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("file locked")
        return real(path, frames)

    monkeypatch.setattr(suite_module, "store_runs", flaky)
    s = BenchmarkSuite(tmp_path / "r.parquet")
    s.add(FakeBenchmark(1))                         # zapis się nie udał, benchmark trwa dalej
    assert "results not written" in capsys.readouterr().out
    s.add(FakeBenchmark(2))                         # zaległy przebieg dopisany razem z nowym
    assert len(s) == 2
    assert sorted(pd.read_parquet(s.out_file)["seed"].unique()) == [1, 2]


# ---------------------------------------------------------------------------------------
# Metryki (analysis/metrics.py)
# ---------------------------------------------------------------------------------------

def test_method_metrics_classification():
    names = ["rf_sklearn", "full_voting", "IB"]
    y_true = np.array([0, 1, 1, 0])
    Y = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 1]])
    N = np.array([[10] * 4, [10] * 4, [2, 4, 6, 10]])
    m = method_metrics(names, y_true, Y, N, np.array([2.0, 4.0, 1.0]), 10, "classification")
    assert m.loc["IB", "accuracy"] == 0.5
    assert m.loc["IB", "agree_full"] == 0.75 and m.loc["IB", "agree_rf"] == 0.5
    assert m.loc["IB", "speedup_vs_full"] == 4.0 and m.loc["IB", "speedup_vs_rf"] == 2.0
    assert m.loc["IB", "trees_mean"] == 5.5 and m.loc["IB", "trees_frac"] == 0.55
    assert m.loc["IB", "ms_per_sample"] == pytest.approx(250.0)
    # bez metody odniesienia nie ma kolumn porównania
    alone = method_metrics(["IB"], y_true, Y[2:], N[2:], np.ones(1), 10, "classification")
    assert not any(c.startswith(("agree_", "speedup_")) for c in alone.columns)


def test_method_metrics_regression_uses_method_delta():
    Y = np.array([[1.0, 2.0, 3.0], [1.5, 2.0, 4.0]])
    m = method_metrics(["full_averaging", "IBREG"], np.array([1.0, 2.0, 3.0]), Y,
                       np.full((2, 3), 10), np.ones(2), 10, "regression", {"IBREG": 0.6})
    assert m.loc["IBREG", "agree_full"] == pytest.approx(2 / 3)
    assert m.loc["IBREG", "mad_full"] == pytest.approx(0.5)
    assert m.loc["IBREG", "maxdiff_full"] == 1.0
    assert m.loc["full_averaging", "agree_full"] == 1.0          # bez delta: np.isclose


def raw_row(method, y_pred, n_used, seed=1, pi=None, **settings):
    """Wiersz pliku wyników z czterema próbkami."""
    return {"dataset": "d", "seed": seed, "T": 10, "task": "classification", **settings,
            "date": "x", "mode": "batch", "method": method, "pi": pi, "delta": None,
            "time_s": 1.0, "y_true": np.array([0, 1, 1, 0]), "y_pred": np.array(y_pred),
            "n_used": np.array(n_used)}


def test_compute_metrics_groups_runs_and_keeps_row_order():
    rows = [raw_row("full_voting", [0, 1, 1, 0], [10] * 4, seed=1),
            raw_row("full_voting", [1, 1, 1, 1], [10] * 4, seed=2),
            raw_row("IB", [0, 1, 1, 1], [3] * 4, seed=2, pi=0.9),
            raw_row("IB", [0, 1, 1, 0], [2] * 4, seed=1, pi=0.9),
            # kolumna, której starsze przebiegi nie mają (NaN), nie gubi grupy
            raw_row("full_voting", [0, 1, 1, 0], [10] * 4, seed=3, val_size=np.nan),
            raw_row("IB", [0, 0, 1, 0], [3] * 4, seed=3, val_size=np.nan)]
    m = compute_metrics(pd.DataFrame(rows, index=[10, 11, 12, 13, 14, 15]))
    assert list(m.index) == [10, 11, 12, 13, 14, 15]
    # agree_full w obrębie przebiegu (seed), a nie między przebiegami
    assert list(m["agree_full"]) == [1.0, 1.0, 0.75, 1.0, 1.0, 0.75]
    assert m.loc[12, "trees_mean"] == 3 and m.loc[13, "trees_mean"] == 2
    with pytest.raises(ValueError, match="same method twice"):
        compute_metrics(pd.DataFrame([rows[0], rows[0]]))


def test_end_to_end_file_to_metrics(tmp_path, binary, regression):
    """Benchmark -> plik Parquet -> metryki: odniesienia zgadzają się same ze sobą."""
    # jeden plik na zadanie, jak w README (etykiety-napisy i liczby nie dzielą kolumny)
    paths = []
    for seed, case, extra in ((1, binary, [IBPruning(pi=0.95), Glee(pi=0.95)]),
                              (2, regression, [IBRegression(delta_std=0.1)])):
        m = ForestMethods(case.rf, X_train=case.X_train, y_train=case.y_train)
        for method in extra:
            m.add_method(method)
        b = Benchmark(m, case.X_test, case.y_test, metadata={"dataset": "d", "seed": seed},
                      n_online=15).run(["batch", "online"])
        paths.append(store_runs(tmp_path / f"{seed}.parquet", [b.release().to_frame()]))
    metrics = compute_metrics(pd.concat([pd.read_parquet(p) for p in paths],
                                        ignore_index=True))
    assert len(metrics) == 2 * 4 + 2 * 3
    full = metrics[metrics["method"].isin(["full_voting", "full_averaging"])]
    assert (full["agree_full"] == 1.0).all() and (full["trees_frac"] == 1.0).all()
    ib = metrics[metrics["method"] == "IB_pi95"]
    assert (ib["trees_frac"] < 1).all() and (ib["agree_full"] > 0.9).all()
    reg = metrics[metrics["task"] == "regression"]
    assert reg["accuracy"].isna().all() and reg["r2"].notna().all()
