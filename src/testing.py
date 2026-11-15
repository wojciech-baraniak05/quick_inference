# %%
"""Benchmark metod inferencji; wszystkie ustawienia są w config.toml."""
import time

from sklearn.model_selection import train_test_split

import config as cfg
from benchmarking import (DATA_DIR, Benchmark, BenchmarkSuite, ForestMethods, check_modes,
                          find_datasets, load_dataset, train_forest)
from methods import build_methods

# %%
MODES = check_modes(cfg.MODES)
if not 0 <= cfg.VAL_SIZE < 1 - cfg.TEST_SIZE:
    raise ValueError(f"val_size must be in [0, 1 - test_size), got {cfg.VAL_SIZE} "
                     f"(test_size = {cfg.TEST_SIZE})")
REFS = Benchmark.REFS
OUT_DIR = DATA_DIR / cfg.OUT_DIR
# walidacja configu; bloki [[methods]] innego zadania niż cfg.TASK są pomijane
names = [m.name for m in build_methods(cfg.METHODS, cfg.SKLEARN_SOFT_VOTING, cfg.TASK)]
datasets = find_datasets(cfg.DATASETS)

print(f"task={cfg.TASK} | {len(datasets)} datasets | seeds={cfg.SEEDS} | "
      f"modes={','.join(MODES)}")
print(f"forest: {cfg.N_TREES} trees, max_depth={cfg.MAX_DEPTH} | "
      f"rf_sklearn={'on' if cfg.SKLEARN_SOFT_VOTING else 'off'} | test_size={cfg.TEST_SIZE}, "
      f"val_size={cfg.VAL_SIZE}, n_online={cfg.N_ONLINE}, n_repeats={cfg.N_REPEATS}")
stems = {p.stem for p in datasets}
for ds, params in cfg.FOREST_PER_DATASET.items():
    # nadpisanie zbioru spoza tego przebiegu nie jest błędem (config bywa wspólny dla kilku
    # list datasets), ale literówka w nazwie byłaby niewidoczna, więc ją pokazujemy
    note = "" if ds in stems else "  (not in datasets, ignored)"
    print(f"forest override {ds}: "
          f"{', '.join(f'{k}={v}' for k, v in cfg.forest_params(ds).items())}{note}")
print(f"methods ({len(names)}): {', '.join(names)}")
print(f"per run [{MODES[0]}]: t = mean trees used, agree = agreement with the full forest, "
      f"x = speedup vs the full forest")

# %%
# out_dir: każdy przebieg jest zapisywany od razu po policzeniu (runs/*.npz + tabele .csv)
suite = BenchmarkSuite(keys=("dataset", "seed"), out_dir=OUT_DIR)
skipped: set[tuple[str, str]] = set()      # (metoda, błąd) już zgłoszone: raz, nie co seed i zbiór

for path in datasets:
    X, y = load_dataset(path, cfg.DROP_COLUMNS)
    forest = cfg.forest_params(path.stem)      # [forest] z nadpisaniami dla tego zbioru

    for seed in cfg.SEEDS:
        start = time.perf_counter()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.TEST_SIZE, random_state=seed)
        X_val = y_val = None
        if cfg.VAL_SIZE > 0:
            # zbiór walidacyjny z danych uczących, odłożony przed treningiem lasu: las go nie
            # widzi; val_size to część całego zbioru, więc tu dzielimy przez 1 - test_size
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=cfg.VAL_SIZE / (1 - cfg.TEST_SIZE),
                random_state=seed)
        rf = train_forest(X_train, y_train, forest["n_trees"], forest["max_depth"], seed,
                          cfg.TASK)

        # nowe instancje metod dla każdego lasu (fit dobiera np. tablicę IB do T i l,
        # a HYPER, MARGIN i metody własne kalibrują się na predykcjach out-of-bag albo,
        # z calib = "val", na zbiorze walidacyjnym)
        m = ForestMethods(rf, methods=[], X_train=X_train, y_train=y_train,
                          X_val=X_val, y_val=y_val)
        for method in build_methods(cfg.METHODS, cfg.SKLEARN_SOFT_VOTING, cfg.TASK):
            try:
                m.add_method(method)
            except (ValueError, FileNotFoundError) as e:
                if (method.name, str(e)) not in skipped:
                    skipped.add((method.name, str(e)))
                    print(f"  skip {method.name} on {path.stem} (shown once): {e}")
        if not m.methods:
            print(f"  skip {path.stem}: no method fits this forest")
            break

        b = Benchmark(m, X_test, y_test,
                      metadata={"dataset": path.stem, "seed": seed, "task": cfg.TASK,
                                "n_trees": forest["n_trees"],
                                "max_depth": forest["max_depth"],
                                "val_size": cfg.VAL_SIZE,
                                "sklearn_soft_voting": cfg.SKLEARN_SOFT_VOTING},
                      n_online=cfg.N_ONLINE, n_repeats=cfg.N_REPEATS)
        b.run(modes=MODES)
        b.release()                 # suite trzyma wyniki do końca, lasu i danych nie musi
        suite.add(b)

        r = b.results(MODES[0])
        parts = []
        for name in r.index.difference(REFS, sort=False):
            w = r.loc[name]
            s = f"{name} t={w.trees_mean:.1f}"
            if "agree_full" in w:
                s += f" agree={w.agree_full:.3f} x{w.speedup_vs_full:.1f}"
            parts.append(s)
        elapsed = time.perf_counter() - start
        print(f"{path.stem:<24} seed={seed:<2} {elapsed:3.0f}s | {' | '.join(parts)}")

# %%
suite.summary()

# %%
# przebiegi, w których zgodność metody z wczesnym stopem z pełnym lasem spadła poniżej jej pi
# (klasyfikacja: ta sama klasa, regresja: różnica w tolerancji delta metody)
pi_levels = {name: pi for b in suite.runs for name, pi in b.pi_levels.items()}
for mode in suite.modes:
    df = suite.results(mode)
    if "agree_full" not in df:
        print("agree_full check skipped: no full_voting / full_averaging in the benchmark")
        break
    counts, failed = [], []
    for name, pi in pi_levels.items():
        agree = df.xs(name, level="method")["agree_full"]
        below = agree[agree < pi]
        counts.append(f"{name} {len(below)}/{len(agree)}")
        if len(below):
            failed.append((name, pi, below))
    print(f"[{mode}] runs with agree_full < pi: {', '.join(counts)}")
    for name, pi, below in failed:
        print(f"  {name} (pi={pi}):\n{below.to_string()}")

# %%
suite.save()                        # przebiegi są już zapisane; to odświeża komplet plików
print(f"saved {len(suite)} runs + tables to {OUT_DIR}")

# %%
