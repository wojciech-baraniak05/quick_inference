# quick_inference

Benchmarks fast inference methods for random forests against full-ensemble prediction.
The main method is **statistical instance-based (IB) pruning**
(Hernández-Lobato, Martínez-Muñoz, Suárez, *IEEE TPAMI* 31(2), 2009).
It queries trees one at a time and stops once the majority vote is settled
with probability ≥ `pi`. It is compared with scikit-learn's `RandomForestClassifier.predict`
and with plain majority voting over all trees.

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

```toml
seed = 1   # random seed for dataset generation and sampling
```

```python
from config import SEED
```

To add a setting, add the key to `config.toml` and expose it in `src/config.py`,
for example `N_TREES = _config["n_trees"]`.

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

## Running the benchmark

[`src/testing.py`](src/testing.py) trains a forest (501 trees, depth 5) on every binary
synthetic dataset for seeds 1–3. It then compares `rf_sklearn`, `full_voting` and `IB_pi98`
in two modes:

- **batch**: the whole test set at once;
- **online**: 500 samples, predicted one by one.

Run it from `src/`, because paths are resolved relative to the working directory:

```sh
cd src
python testing.py
```

You can also run its `# %%` cells interactively in VS Code.
Results are written to `data/results/binary/`.

Reported per method: accuracy, macro precision/recall/F1, agreement with sklearn and with
full voting (`agree_rf`, `agree_full`), prediction time, speed-up, and trees queried per sample
(`trees_mean`, `trees_median`, `trees_max`, `trees_frac`).

### Adding an inference method

Subclass `InferenceMethod` in [`src/methods.py`](src/methods.py) and implement:

- `predict_batch(X) -> (y_pred, n_used)`
- `predict_one(x) -> (y, n_used)`

Here `n_used` is the number of trees queried. Then register the method with the benchmark:

```python
m = methods_rf(rf)              # includes rf_sklearn and full_voting
m.add_method(MyMethod())
```

## Project layout

```
config.toml                  project settings (seed)
setup_windows.bat            venv + requirements + datasets
requirements.txt
data/
  generated/                 synthetic datasets (generated, git-ignored)
  real_data/                 UCI datasets (downloaded, git-ignored)
  tablice/                   IB pruning stop tables
  results/                   benchmark output (git-ignored)
src/
  config.py                  reads config.toml
  methods.py                 inference methods, benchmark, benchmark_suite
  generate_table_IB.py       stop-table generation for IB pruning
  testing.py                 benchmark run over the binary datasets
  test_dt.ipynb              scratch notebook for forest/tree experiments
  data_generate/
    generate_datasets.py     synthetic dataset generation
    data_download_real.py    real dataset download
  analysis/                  exploratory notebooks (real datasets, generated configurations)
```
