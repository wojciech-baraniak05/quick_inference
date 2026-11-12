# %%
"""Benchmark metod inferencji; wszystkie ustawienia są w config.toml."""
import time

from sklearn.model_selection import train_test_split

import config as cfg
from benchmarking import (DATA_DIR, benchmark, benchmark_suite, check_modes, methods_rf,
                          train_forest, wczytaj, znajdz_zbiory)
from methods import build_methods

# %%
MODES = check_modes(cfg.MODES)
REFS = (benchmark.REF_RF, benchmark.REF_FULL)
OUT_DIR = DATA_DIR / cfg.OUT_DIR
nazwy = [m.name for m in build_methods(cfg.METHODS, cfg.SKLEARN_SOFT_VOTING)]  # walidacja configu
datasets = znajdz_zbiory(cfg.DATASETS)

print(f"{len(datasets)} zbiorów, seedy {cfg.SEEDS}, scenariusze {list(MODES)}")
print(f"las: {cfg.N_TREES} drzew, max_depth={cfg.MAX_DEPTH}, "
      f"soft voting sklearn: {'tak' if cfg.SKLEARN_SOFT_VOTING else 'nie'}")
print(f"metody: {nazwy}")

# %%
# out_dir: każdy przebieg jest zapisywany od razu po policzeniu (runs/*.npz + tabele .csv)
suite = benchmark_suite(keys=("dataset", "seed"), out_dir=OUT_DIR)

for plik in datasets:
    X, y = wczytaj(plik)

    for seed in cfg.SEEDS:
        start = time.perf_counter()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.TEST_SIZE, random_state=seed)
        rf = train_forest(X_train, y_train, cfg.N_TREES, cfg.MAX_DEPTH, seed)

        # nowe instancje metod dla każdego lasu (fit dobiera np. tablicę IB do T i l,
        # a HYPER i MARGIN kalibrują się na głosach out-of-bag danych uczących)
        m = methods_rf(rf, methods=[], X_train=X_train, y_train=y_train)
        for method in build_methods(cfg.METHODS, cfg.SKLEARN_SOFT_VOTING):
            try:
                m.add_method(method)
            except (ValueError, FileNotFoundError) as e:
                print(f"  {plik.stem}: pomijam {method.name}: {e}")
        if not m.methods:
            print(f"  {plik.stem}: żadna metoda nie pasuje do lasu, pomijam zbiór")
            break

        b = benchmark(m, X_test, y_test,
                      metadata={"dataset": plik.stem, "seed": seed,
                                "n_trees": cfg.N_TREES, "max_depth": cfg.MAX_DEPTH,
                                "sklearn_soft_voting": cfg.SKLEARN_SOFT_VOTING},
                      n_online=cfg.N_ONLINE, n_repeats=cfg.N_REPEATS)
        b.run(modes=MODES)
        b.release()                 # suite trzyma wyniki do końca, lasu i danych nie musi
        suite.add(b)

        r = b.results(MODES[0])
        opis = []
        for name in r.index.difference(REFS, sort=False):
            w = r.loc[name]
            s = f"{name}: trees={w.trees_mean:.1f}"
            if "agree_full" in w:
                s += f" agree_full={w.agree_full:.4f} x{w.speedup_vs_full:.1f}"
            opis.append(s)
        print(f"{plik.stem:24s} seed={seed}  {'  '.join(opis)}  "
              f"({time.perf_counter() - start:.0f} s)")

# %%
suite.summary()

# %%
# przebiegi, w których zgodność metody z wczesnym stopem (IB, HYPER, GLEE) z pełnym
# głosowaniem spadła poniżej jej pi
poziomy = {name: pi for b in suite.runs for name, pi in b.pi_levels.items()}
for mode in suite.modes:
    df = suite.results(mode)
    if "agree_full" not in df:
        print("brak full_voting w benchmarku - pomijam sprawdzenie agree_full")
        break
    for name, pi in poziomy.items():
        agree = df.xs(name, level="method")["agree_full"]
        ponizej = agree[agree < pi]
        print(f"[{mode}] {name}: agree_full < {pi}: {len(ponizej)} z {len(agree)} przebiegów")
        if len(ponizej):
            print(ponizej.to_string())

# %%
suite.save()                        # przebiegi są już zapisane; to odświeża komplet plików
print(f"zapisano wyniki do {OUT_DIR}")

# %%
