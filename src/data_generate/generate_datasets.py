import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import make_classification
def main():
    load_dotenv()
    SEED = int(os.environ["SEED"])
    DATA_DIR = Path(__file__).resolve().parents[2] / "data"
    DATASETS = {
        "binary_small": dict(n_samples=1_000, n_features=10, n_informative=5, n_redundant=2, n_classes=2),
        "binary_large": dict(n_samples=500_000, n_features=20, n_informative=10, n_redundant=5, n_classes=2),
        "imbalanced": dict(n_samples=10_000, n_features=20, n_informative=8, n_redundant=4, n_classes=2, weights=[0.95, 0.05]),
        "binary_check": dict(n_samples=100_000, n_features=10, n_informative=5, n_redundant=2, n_classes=2),
        "binary_multi_features": dict(n_samples=50_000, n_features=100, n_informative=50, n_redundant=10, n_classes=2),
    }
    CATEGORICAL_FEATURES = {"binary_multi_features": 20}
    CATEGORY_LABELS = ["a", "b", "c", "d", "e"]
    DATA_DIR.mkdir(exist_ok=True)
    for name, params in DATASETS.items():
        X, y = make_classification(**params, random_state=SEED)
        df = pd.DataFrame(X).add_prefix("f")

        for col in df.columns[:CATEGORICAL_FEATURES.get(name, 0)]:
            df[col] = pd.qcut(df[col], q=len(CATEGORY_LABELS), labels=CATEGORY_LABELS)

        df["target"] = y
        df.to_parquet(DATA_DIR / f"{name}.parquet", index=False)

        print(f"Saved {name}.parquet: {X.shape[0]} samples, {X.shape[1]} features")

if __name__ == "__main__":
    main()