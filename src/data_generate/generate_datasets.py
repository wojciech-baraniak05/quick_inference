"""Generuje syntetyczne zbiory danych (.parquet) do data/generated/."""
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_friedman1, make_regression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SEED  # noqa: E402


Params = dict[str, Any]         # argumenty generatora scikit-learn dla jednego zbioru


def main() -> None:
    DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "generated"
    BINARY: dict[str, Params] = {
        "binary_small": dict(n_samples=1_000, n_features=10, n_informative=5, n_redundant=2, n_classes=2, flip_y=0.02),
        "binary_large": dict(n_samples=500_000, n_features=20, n_informative=10, n_redundant=5, n_classes=2, flip_y=0.02),
        "imbalanced": dict(n_samples=10_000, n_features=20, n_informative=8, n_redundant=4, n_classes=2, weights=[0.95, 0.05], flip_y=0.02),
        "binary_check": dict(n_samples=100_000, n_features=5, n_informative=5, n_redundant=0, n_classes=2, flip_y=0.02),
        "binary_multi_features": dict(n_samples=50_000, n_features=100, n_informative=50, n_redundant=10, n_classes=2, flip_y=0.02),
        "noisy_labels": dict(n_samples=20_000, n_features=20, n_informative=10, n_redundant=4, n_classes=2, flip_y=0.25),
        "hard_nonlinear": dict(n_samples=20_000, n_features=10, n_informative=6, n_redundant=0, n_classes=2, n_clusters_per_class=6, class_sep=0.5),
        "wide_p_gt_n": dict(n_samples=500, n_features=2_000, n_informative=15, n_redundant=5, n_classes=2),
        "rare_event": dict(n_samples=200_000, n_features=30, n_informative=10, n_redundant=5, n_classes=2, weights=[0.995, 0.005], flip_y=0.0),
        "collinear_unscaled": dict(n_samples=20_000, n_features=40, n_informative=8, n_redundant=24, n_repeated=4, n_classes=2, shift=None, scale=None),
        "outliers": dict(n_samples=20_000, n_features=20, n_informative=10, n_redundant=4, n_classes=2, flip_y=0.02),
    }
    MULTICLASS: dict[str, Params] = {
        "multi_small": dict(n_samples=1_000, n_features=10, n_informative=5, n_redundant=2, n_classes=4, flip_y=0.02),
        "multi_large": dict(n_samples=500_000, n_features=20, n_informative=10, n_redundant=5, n_classes=5, flip_y=0.02),
        "multi_imbalanced": dict(n_samples=10_000, n_features=20, n_informative=8, n_redundant=4, n_classes=4, weights=[0.7, 0.2, 0.07, 0.03], flip_y=0.02),
        "multi_check": dict(n_samples=100_000, n_features=5, n_informative=5, n_redundant=0, n_classes=3, flip_y=0.02),
        "multi_multi_features": dict(n_samples=50_000, n_features=100, n_informative=50, n_redundant=10, n_classes=5, flip_y=0.02),
        "multi_many_classes": dict(n_samples=50_000, n_features=30, n_informative=15, n_redundant=5, n_classes=20, n_clusters_per_class=1, flip_y=0.02),
        "multi_noisy_labels": dict(n_samples=20_000, n_features=20, n_informative=10, n_redundant=4, n_classes=5, flip_y=0.25),
        "multi_hard_nonlinear": dict(n_samples=20_000, n_features=10, n_informative=6, n_redundant=0, n_classes=4, n_clusters_per_class=6, class_sep=0.5),
        "multi_wide_p_gt_n": dict(n_samples=500, n_features=2_000, n_informative=15, n_redundant=5, n_classes=4),
        "multi_rare_class": dict(n_samples=200_000, n_features=30, n_informative=10, n_redundant=5, n_classes=3, weights=[0.95, 0.045, 0.005], flip_y=0.0),
        "multi_collinear_unscaled": dict(n_samples=20_000, n_features=40, n_informative=8, n_redundant=24, n_repeated=4, n_classes=4, shift=None, scale=None),
        "multi_outliers": dict(n_samples=20_000, n_features=20, n_informative=10, n_redundant=4, n_classes=4, flip_y=0.02),
    }
    REGRESSION: dict[str, Params] = {
        "reg_small": dict(n_samples=1_000, n_features=10, n_informative=5, noise=10.0),
        "reg_large": dict(n_samples=500_000, n_features=20, n_informative=10, noise=10.0),
        "reg_check": dict(n_samples=100_000, n_features=5, n_informative=5, noise=1.0),
        "reg_multi_features": dict(n_samples=50_000, n_features=100, n_informative=50, noise=10.0),
        "reg_noisy": dict(n_samples=20_000, n_features=20, n_informative=10, noise=100.0),
        "reg_wide_p_gt_n": dict(n_samples=500, n_features=2_000, n_informative=15, noise=5.0),
        # effective_rank: macierz cech niskiego rzędu -> silnie skorelowane cechy
        "reg_collinear": dict(n_samples=20_000, n_features=40, n_informative=8, effective_rank=5, tail_strength=0.1, noise=5.0),
        "reg_outliers": dict(n_samples=20_000, n_features=20, n_informative=10, noise=10.0),
    }
    # make_regression jest liniowe, więc nieliniową zależność daje Friedman #1 (używa tylko pierwszych 5 cech)
    NONLINEAR_REGRESSION: dict[str, Params] = {
        "reg_hard_nonlinear": dict(n_samples=20_000, n_features=10, noise=1.0),
    }
    # (generator, zbiory, podfolder w DATA_DIR)
    GENERATORS: list[tuple[Callable[..., Any], dict[str, Params], str]] = [
        (make_classification, BINARY, "binary"),
        (make_classification, MULTICLASS, "multiclass"),
        (make_regression, REGRESSION, "regression"),
        (make_friedman1, NONLINEAR_REGRESSION, "regression"),
    ]
    # fraction: odsetek wierszy z outlierami, scale: siła zaburzenia w odchyleniach std cechy
    OUTLIERS: dict[str, dict[str, float]] = {
        "outliers": dict(fraction=0.02, scale=10.0),
        "multi_outliers": dict(fraction=0.02, scale=10.0),
        "reg_outliers": dict(fraction=0.02, scale=10.0),
    }
    for generator, datasets, subdir in GENERATORS:
        out_dir = DATA_DIR / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, params in datasets.items():
            X, y = generator(**params, random_state=SEED)

            if name in OUTLIERS:
                cfg = OUTLIERS[name]
                rng = np.random.default_rng(SEED)
                rows = rng.choice(len(X), size=int(cfg["fraction"] * len(X)), replace=False)
                X[rows] += rng.normal(0, cfg["scale"] * X.std(axis=0), size=(len(rows), X.shape[1]))

            df = pd.DataFrame(X).add_prefix("f")
            df["target"] = y
            df.to_parquet(out_dir / f"{name}.parquet", index=False)

            print(f"Saved {subdir}/{name}.parquet: {X.shape[0]} samples, {X.shape[1]} features")


if __name__ == "__main__":
    main()
