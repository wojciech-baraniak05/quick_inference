# Data flow of one benchmark run

UML sequence diagram of `src/testing.py` for the configuration currently in `config.toml`:
`task = "classification"`, `datasets = ["generated/binary"]`, `seeds = [1, 2, 3]`,
`modes = ["batch", "online"]`, a forest of 501 trees with `max_depth = 5`, and seven enabled
methods (`rf_sklearn`, `full_voting`, `IB_pi98`, `HYPER_pi98`, `GLEE_pi98`, `MARGIN_k0`,
`OWN_lam0.5_pi98`).

```mermaid
sequenceDiagram
  participant T as testing.py
  participant R as methods<br/>registry
  participant D as benchmarking<br/>data
  participant F as methods_rf
  participant M as Inference<br/>Method x7
  participant B as benchmark
  participant S as benchmark<br/>suite

  Note over T: import config<br/>config.py parses config.toml with tomllib<br/>into TASK, DATASETS, SEEDS, MODES,<br/>N_TREES, MAX_DEPTH, METHODS ...
  T->>R: build_methods(METHODS, SKLEARN_SOFT_VOTING, TASK)
  R-->>T: 7 unfitted methods (config validation)
  T->>D: znajdz_zbiory(DATASETS)
  D-->>T: paths of data/generated/binary/*.parquet
  T->>S: benchmark_suite(keys = dataset, seed, out_dir = data/results/binary)

  loop every .parquet file
    T->>D: wczytaj(plik, DROP_COLUMNS)
    D-->>T: X, y
    loop every seed in 1, 2, 3
      T->>T: train_test_split(test_size = 0.3, seed)
      T->>D: train_forest(X_train, y_train, 501, 5, seed, TASK)
      D-->>T: rf (RandomForestClassifier)
      T->>F: methods_rf(rf, X_train, y_train)
      T->>R: build_methods(...)
      R-->>T: fresh method instances
      loop every method
        T->>F: add_method(method)
        F->>M: fit(rf, X_train, y_train)
        Note over M: IB reads data/tablice/T501_L2_pi98.npz<br/>HYPER, OWN, MARGIN calibrate on out-of-bag votes<br/>GLEE computes its table from T and pi
        M-->>F: fitted method with stop thresholds
      end
      T->>B: benchmark(m, X_test, y_test, metadata, N_ONLINE, N_REPEATS)
      T->>B: run(MODES)
      loop mode in batch, online
        B->>F: run(first n rows of X_test, mode, n_repeats)
        F->>M: predict_batch(X) or predict_one(x)
        M-->>F: y_pred, n_used
        F-->>B: Y_pred, N_used, times
      end
      T->>B: release()
      T->>S: add(b)
      S->>B: save(runs/DATASET_seedN.npz), results(mode)
      B-->>S: metrics table, one row per method
      Note over S: writes results_MODE.csv and aggregate_MODE.csv<br/>to data/results/binary
    end
  end

  T->>S: summary(), results(mode), save()
  S-->>T: aggregated tables for the agree_full vs pi check
```

## What travels along the arrows

| Data | Produced by | Consumed by |
|---|---|---|
| Constants (`TASK`, `DATASETS`, `SEEDS`, `MODES`, `N_TREES`, `MAX_DEPTH`, `METHODS`, ...) | `config.py` from `config.toml` | `testing.py` only; `methods` and `benchmarking` never read the config |
| `list[InferenceMethod]` (unfitted) | `registry.build_methods`: one `[[methods]]` block becomes one constructor call, `rf_sklearn` is prepended when `sklearn_soft_voting = true`, blocks with `enabled = false` or of the other task are skipped | `methods_rf.add_method` |
| `X, y` | `data.wczytaj` from a `.parquet` file | `train_test_split` |
| `rf` | `data.train_forest` on `X_train, y_train` | `methods_rf`, then every method's `fit` |
| Stop thresholds (`prog`, `voters`) | each method's `fit`: IB from a `.npz` table, HYPER/OWN/MARGIN from out-of-bag predictions on `X_train`, GLEE from `T` and `pi` | `predict_batch` / `predict_one` |
| `Y_pred, N_used, times` (one row per method) | `methods_rf.run` on `X_test` | `benchmark._compute_results` |
| Metrics table (accuracy, f1, `agree_*`, `speedup_vs_*`, `trees_*`) | `benchmark.results(mode)` using `y_test` | `benchmark_suite`, which writes `runs/*.npz`, `results_*.csv`, `aggregate_*.csv` |
