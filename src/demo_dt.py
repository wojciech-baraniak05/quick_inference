import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from decision_tree import DecisionTreeClassifier


def main():
    load_dotenv()
    SEED = int(os.environ["SEED"])
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"

    df = pd.read_parquet(DATA_DIR / "binary_small.parquet")
    X = df.drop(columns="target").to_numpy()
    y = df["target"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2).fit(X_train, y_train)

    pred = tree.predict(X_test)
    proba = tree.predict_proba(X_test)
    leaves = tree.apply(X_test)
    print(f"accuracy: {np.mean(pred == y_test):.4f}")
    print(f"first 5 predictions: {pred[:5]}")
    print(f"first 5 probabilities:\n{proba[:5]}")
    print(f"first 5 leaf ids: {leaves[:5]}")

    print(f"\nnodes: {len(tree.feature)}, leaves: {np.sum(tree.children_left == -1)}, classes: {tree.classes_}")
    print(f"{'node':>4} {'left':>5} {'right':>5} {'feature':>7} {'threshold':>9}  value")
    for i in range(5):
        print(f"{i:>4} {tree.children_left[i]:>5} {tree.children_right[i]:>5} {tree.feature[i]:>7} {tree.threshold[i]:>9.4f}  {tree.value[i].round(3)}")


if __name__ == "__main__":
    main()
