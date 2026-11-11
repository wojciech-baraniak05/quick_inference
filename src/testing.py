# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from methods import IBPruning, benchmark, benchmark_suite, methods_rf

# %%
path = Path.cwd().parent / "data"
DATA_DIR = path / "generated" / "binary"
TABLE = path / "tablice" / "T501_L2_pi98.npz"
OUT_DIR = path / "results" / "binary"

N_SEEDS = 3          # powtórzenie r używa seed = r (1, 2, 3)
N_TREES = 501
MAX_DEPTH = 5
N_ONLINE = 500       # liczba próbek w scenariuszu online
N_REPEATS = 1        # ile razy mierzyć czas każdej metody (mediana)

tablica = np.load(TABLE)
prog, pi = tablica["prog"], float(tablica["pi"])
datasets = sorted(DATA_DIR.glob("*.parquet"))
print(f"{len(datasets)} zbiorów, seedy 1..{N_SEEDS}, tablica {TABLE.name}")


def wczytaj(plik):
    """Zwraca (X, y); kolumny kategoryczne/tekstowe zamieniane na kody całkowite."""
    df = pd.read_parquet(plik)
    cechy = df.drop(columns="target")
    for c in cechy.select_dtypes(exclude="number").columns:
        cechy[c] = cechy[c].astype("category").cat.codes
    return cechy.to_numpy(dtype=float), df["target"].to_numpy()


# %%
suite = benchmark_suite(keys=("dataset", "seed"))

for plik in datasets:
    X, y = wczytaj(plik)

    for seed in range(1, N_SEEDS + 1):
        start = time.perf_counter()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed)

        # trening równolegle, predykcja w jednym wątku (uczciwe porównanie czasów)
        rf = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                    random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf.set_params(n_jobs=None)

        m = methods_rf(rf)                                   # rf_sklearn + full_voting
        m.add_method(IBPruning(prog, name="IB_pi98", pi=pi))

        b = benchmark(m, X_test, y_test,
                      metadata={"dataset": plik.stem, "seed": seed, "pi": pi},
                      n_online=N_ONLINE, n_repeats=N_REPEATS)
        b.run()
        suite.add(b)

        r = b.results("batch").loc["IB_pi98"]
        print(f"{plik.stem:24s} seed={seed}  agree_full={r.agree_full:.4f}  "
              f"trees={r.trees_mean:6.1f}  speedup_vs_full=x{r.speedup_vs_full:.1f}  "
              f"({time.perf_counter() - start:.0f} s)")

# %%
suite.summary()

# %%
# przebiegi, w których zgodność IB z pełnym głosowaniem spadła poniżej pi
for mode in ("batch", "online"):
    agree = suite.results(mode).xs("IB_pi98", level="method")["agree_full"]
    ponizej = agree[agree < pi]
    print(f"[{mode}] agree_full < {pi}: {len(ponizej)} z {len(agree)} przebiegów")
    if len(ponizej):
        print(ponizej.to_string())

# %%
suite.save(OUT_DIR)
print(f"zapisano wyniki do {OUT_DIR}")

# %%
