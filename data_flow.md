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
  participant F as ForestMethods
  participant M as Inference<br/>Method x7
  participant B as benchmark
  participant S as benchmark<br/>suite

  Note over T: import config<br/>config.py parses config.toml with tomllib<br/>into TASK, DATASETS, SEEDS, MODES,<br/>N_TREES, MAX_DEPTH, METHODS ...
  T->>R: build_methods(METHODS, SKLEARN_SOFT_VOTING, TASK)
  R-->>T: 7 unfitted methods (config validation)
  T->>D: find_datasets(DATASETS)
  D-->>T: paths of data/generated/binary/*.parquet
  T->>S: BenchmarkSuite(keys = dataset, seed, out_file = data/results/binary.parquet)

  loop every .parquet file
    T->>D: load_dataset(path, DROP_COLUMNS)
    D-->>T: X, y
    loop every seed in 1, 2, 3
      T->>T: train_test_split(test_size = 0.3, seed)
      opt val_size > 0
        T->>T: train_test_split(X_train, val_size / (1 - test_size), seed)
        Note over T: X_val, y_val held out of training<br/>(otherwise None)
      end
      T->>D: train_forest(X_train, y_train, 501, 5, seed, TASK)
      D-->>T: rf (RandomForestClassifier)
      T->>F: ForestMethods(rf, X_train, y_train, X_val, y_val)
      T->>R: build_methods(...)
      R-->>T: fresh method instances
      loop every method
        T->>F: add_method(method)
        F->>M: fit(rf, X_train, y_train, X_val, y_val)
        Note over M: IB reads data/tablice/T501_L2_pi98.npz<br/>HYPER, OWN, MARGIN calibrate on out-of-bag votes<br/>(calib = "val": on X_val, y_val)<br/>GLEE computes its table from T and pi
        M-->>F: fitted method with stop thresholds
      end
      T->>B: Benchmark(m, X_test, y_test, metadata, N_ONLINE, N_REPEATS)
      T->>B: run(MODES)
      loop mode in batch, online
        B->>F: run(first n rows of X_test, mode, n_repeats)
        F->>M: predict_batch(X) or predict_one(x)
        M-->>F: y_pred, n_used
        F-->>B: Y_pred, N_used, times
      end
      T->>B: release()
      T->>S: add(b)
      S->>B: to_frame()
      B-->>S: raw rows, one per mode and method
      Note over S: store_runs: adds the rows to data/results/binary.parquet<br/>(same settings, mode and method replace the stored row)
      T->>B: results(MODES[0])
      B-->>T: metrics table for the progress line (metrics.method_metrics)
    end
  end

  T->>S: summary(), results(mode), save()
  S-->>T: aggregated tables for the agree_full vs pi check
```

## What travels along the arrows

| Data | Produced by | Consumed by |
|---|---|---|
| Constants (`TASK`, `DATASETS`, `SEEDS`, `MODES`, `N_TREES`, `MAX_DEPTH`, `METHODS`, ...) | `config.py` from `config.toml` | `testing.py` only; `methods` and `benchmarking` never read the config |
| `list[InferenceMethod]` (unfitted) | `registry.build_methods`: one `[[methods]]` block becomes one constructor call, `rf_sklearn` is prepended when `sklearn_soft_voting = true`, blocks with `enabled = false` or of the other task are skipped | `ForestMethods.add_method` |
| `X, y` | `data.load_dataset` from a `.parquet` file | `train_test_split` |
| `X_val, y_val` | a second `train_test_split` of the training part when `val_size > 0` (`None` otherwise); the forest never sees them | `fit` of methods with `calib = "val"` |
| `rf` | `data.train_forest` on `X_train, y_train` | `ForestMethods`, then every method's `fit` |
| Stop thresholds (`threshold`, `voters`) | each method's `fit`: IB from a `.npz` table, HYPER/OWN/MARGIN from out-of-bag predictions on `X_train` (or, with `calib = "val"`, predictions on `X_val`), GLEE from `T` and `pi` | `predict_batch` / `predict_one` |
| `Y_pred, N_used, times` (one row per method) | `ForestMethods.run` on `X_test` | `Benchmark.results` and `Benchmark.to_frame` |
| Raw rows (settings, `pi`, `delta`, `time_s`, `y_true`, `y_pred`, `n_used`) | `Benchmark.to_frame()` | `store_runs`, which writes them to the results `.parquet` |
| Metrics table (accuracy, f1, `agree_*`, `speedup_vs_*`, `trees_*`) | `metrics.method_metrics`: in memory via `Benchmark.results(mode)`, from the file via `compute_metrics` | `BenchmarkSuite.summary()`, the progress line and the `agree_full` vs `pi` check in `testing.py`; analysis |
