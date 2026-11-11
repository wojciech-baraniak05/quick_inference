# quick_inference

Benchmarks fast inference methods for random forests against full-ensemble prediction.
The main method is **statistical instance-based (IB) pruning**
(Hernández-Lobato, Martínez-Muñoz, Suárez, *IEEE TPAMI* 31(2), 2009).
It queries trees one at a time and stops once the majority vote is settled
with probability ≥ `pi`. It is compared with scikit-learn's `RandomForestClassifier.predict`
and with plain majority voting over all trees.

A second early-stopping method, **HYPER** (Soto, Suárez, Martínez-Muñoz, *An urn model for
majority voting in classification ensembles*, NIPS 2016), uses the same stopping rule but
replaces IB pruning's uniform prior over vote outcomes with a prior estimated from the
forest's out-of-bag votes (see [HYPER](#hyper-data-dependent-prior)).

## How to run?

Requires Python 3.11 or newer. `requirements.txt` is pinned against Python 3.15.

### Windows: one script

```bat
setup_windows.bat
```

The script:

1. creates a virtual environment in `.venv` (or reuses an existing one),
2. installs `requirements.txt`,
3. generates the synthetic datasets into `data/generated/`,
4. downloads the real datasets into `data/real_data/`.

Step 4 is skipped if the real datasets are already present.
Run `setup_windows.bat --force-download` to download them again.
SUSY is large, so the first download takes a while.

Then activate the environment:

```bat
.venv\Scripts\activate
```

### Manual (any OS)

1. Set up your venv:
   ```sh
   python -m venv .venv
   # Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
   ```
2. Install the requirements:
   ```sh
   pip install -r requirements.txt
   ```
3. Generate and download the datasets:
   ```sh
   python src/data_generate/generate_datasets.py
   python src/data_generate/data_download_real.py
   ```

## Configuration

Settings live in [`config.toml`](config.toml) in the repository root.
[`src/config.py`](src/config.py) reads the file and exposes the values as module constants:

```python
from config import SEED, MODES, METHODS
```

| Section | Keys |
|---|---|
| top level | `seed`: random seed for synthetic dataset generation |
| `[benchmark]` | `datasets` (folders or files under `data/`), `out_dir`, `modes` (`batch`, `online`), `seeds`, `test_size`, `n_online`, `n_repeats`, `sklearn_soft_voting` |
| `[forest]` | `n_trees`, `max_depth` (omit for unlimited depth) |
| `[[methods]]` | one block per benchmarked method: `type` (`full_voting`, `ib_pruning` or `hyper`), optional `name` and `enabled`, plus the method's parameters |

`sklearn_soft_voting = true` adds `rf_sklearn` to the benchmark. That is scikit-learn's
`rf.predict`, which averages tree probabilities (soft voting), and the results get the
`agree_rf` and `speedup_vs_rf` columns.
With `false`, the benchmark only runs methods that query individual trees (hard voting).

An `ib_pruning` method takes `pi`, and its table `T{n_trees}_L{l}_pi{pi*100}.npz` is
picked to match each forest. You can give a specific file with `table` instead.
If no table exists for a dataset's number of classes, the method is skipped for that
dataset with a warning.

A `hyper` method takes `pi` and optionally `n_prior` (number of training samples used for
the out-of-bag votes, default 20000) and `smooth` (prior smoothing window, default 5).
It needs no table file.

To add a setting, add the key to `config.toml` and expose it in `src/config.py`.

## Datasets

**Synthetic** (`src/data_generate/generate_datasets.py`, seeded by `seed`), saved as Parquet:

| Folder | Count | Variants |
|---|---|---|
| `data/generated/binary/` | 11 | small/large, imbalanced, rare event, noisy labels, hard non-linear, p > n, collinear/unscaled, outliers, many features |
| `data/generated/multiclass/` | 12 | the same variants, plus many classes (20) |
| `data/generated/regression/` | 9 | small/large, noisy, p > n, collinear, outliers, many features, non-linear (Friedman #1) |

**Real** (`src/data_generate/data_download_real.py`, downloaded from the UCI repository):

| File | Task | Source |
|---|---|---|
| `data/real_data/binary/susy.parquet` | binary | SUSY |
| `data/real_data/multiclass/shuttle.parquet` | multiclass | Statlog (Shuttle) |
| `data/real_data/regression/gas_turbine.parquet` | regression (target `NOX`; `year` and `CO` kept as extra columns) | Gas Turbine CO and NOx Emission |

Every file has feature columns `f0, f1, …` and a `target` column.

## IB pruning stop tables

IB pruning uses a precomputed table of stopping thresholds. The table depends only on
the number of trees `T`, the number of classes `l` and the confidence level `pi`.
Tables are stored in `data/tablice/` as `T{T}_L{l}_pi{pi*100}.npz`.
`T501_L2_pi98.npz`, `T501_L3_pi98.npz` and `T51_L2_pi95.npz` are already included.

To generate another table, run this from `src/`:

```python
from generate_table_IB import generuj_tablice
generuj_tablice(T=501, l=2, pi=0.98)
```

`T` must equal the forest's `n_estimators` and `l` its number of classes.
Otherwise `IBPruning.fit` raises an error.

## HYPER: data-dependent prior

For one sample the forest is an urn of `T` marbles (trees) in `l` colours (classes) with
unknown composition; querying trees is drawing without replacement. Given the votes `t`
seen so far, the posterior of the composition `T` is `P(T|t) ∝ ∏ C(T_i, t_i) · P(T)`, and
voting stops when the posterior probability that the full forest elects the current leader
is ≥ `pi`. With a uniform `P(T)` this is exactly IB pruning; HYPER estimates `P(T)` instead:

1. `fit(rf, X_train)` collects the out-of-bag votes of the training samples, rescales each
   to `T` trees and builds a histogram of compositions, smoothed with a window of 5;
2. a backward recursion over vote counts turns the prior into one threshold table per
   leading class (classes are no longer interchangeable);
3. prediction is the same sequential loop as IB pruning with a different table lookup.

The table is computed in `fit` (about 0.03 s for `T=501, l=2`, 7 s for `T=501, l=3`) and is
not stored. Forests with more than 2,000,000 prior cells `(T+1)^(l-1)` are rejected, so with
501 trees HYPER works for 2 and 3 classes. `Hyper.expected_trees()` gives the paper's
Monte Carlo estimate of the mean number of queried trees from the prior alone.

Because the prior is fitted to the data, HYPER's `agree_full` lands close to `pi` rather
than well above it, and individual runs can fall slightly below `pi`.

## Running the benchmark

[`src/testing.py`](src/testing.py) reads everything from `config.toml`.
For every dataset in `benchmark.datasets` and every seed in `benchmark.seeds`, it trains a forest
(`[forest]`) and runs the selected methods (`sklearn_soft_voting` and `[[methods]]`) in the
selected `modes`:

- **batch**: the whole test set at once;
- **online**: the first `n_online` samples, predicted one by one.

The default config compares `rf_sklearn`, `full_voting`, `IB_pi98` and `HYPER_pi98` on the binary synthetic
datasets (501 trees, depth 5, seeds 1–3).

Run it from `src/`:

```sh
cd src
python testing.py
```

You can also run its `# %%` cells interactively in VS Code.
Results are written to `data/<benchmark.out_dir>/`.

Reported per method: accuracy, macro precision/recall/F1, agreement with sklearn and with
full voting (`agree_rf`, `agree_full`), prediction time, speed-up, and trees queried per sample
(`trees_mean`, `trees_median`, `trees_max`, `trees_frac`).
The `*_rf` and `*_full` columns appear only when `rf_sklearn` and `full_voting` are in the benchmark.

### Adding an inference method

Create a module in [`src/methods/`](src/methods/) with a subclass of `InferenceMethod`
(from `methods/base.py`) and implement:

- `predict_batch(X) -> (y_pred, n_used)`
- `predict_one(x) -> (y, n_used)`

An early-stopping method only needs a stopping rule: subclass `SequentialPruning` and
implement `_stop_batch(votes) -> bool mask` (the query loops are shared by IB pruning and HYPER).

Here `n_used` is the number of trees queried. To make the method selectable in `config.toml`,
add it to `METHOD_TYPES` in [`src/methods/registry.py`](src/methods/registry.py):

```python
METHOD_TYPES = {..., "my_method": MyMethod}
```

```toml
[[methods]]
type = "my_method"
some_param = 3        # passed to MyMethod(some_param=3)
```

Without the config, you can also use the method directly:

```python
from benchmarking import methods_rf
m = methods_rf(rf)              # includes rf_sklearn and full_voting
                                # pass X_train=... for methods that need training data (hyper)
m.add_method(MyMethod())
```

## Project layout

```
config.toml                  project settings (datasets, benchmark, forest, methods)
setup_windows.bat            venv + requirements + datasets
requirements.txt
data/
  generated/                 synthetic datasets (generated, git-ignored)
  real_data/                 UCI datasets (downloaded, git-ignored)
  tablice/                   IB pruning stop tables
  results/                   benchmark output (git-ignored)
src/
  config.py                  reads config.toml
  methods/                   inference methods, one module each
    base.py                  InferenceMethod interface, SequentialPruning (early-stopping loop)
    voting.py                rf_sklearn (soft voting), full_voting (hard voting)
    ib_pruning.py            IB pruning (uniform prior, table from file)
    hyper.py                 HYPER (out-of-bag prior, table computed in fit)
    registry.py              builds methods from [[methods]] in config.toml
  benchmarking/              benchmark machinery, independent of config.toml
    modes.py                 batch / online scenarios
    forest_methods.py        methods_rf: methods fitted to one forest
    benchmark.py             benchmark: one forest, one test set
    suite.py                 benchmark_suite: many datasets and seeds
    data.py                  dataset loading, forest training
  generate_table_IB.py       stop-table generation for IB pruning
  testing.py                 benchmark run driven by config.toml
  test_dt.ipynb              scratch notebook for forest/tree experiments
  data_generate/
    generate_datasets.py     synthetic dataset generation
    data_download_real.py    real dataset download
  analysis/                  exploratory notebooks (real datasets, generated configurations)
```
