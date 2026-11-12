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

A third one, **GLEE** (Gaussian Lazy Ensemble Evaluation, from Basilico, Munson, Kolda,
Dixon, Kegelmeyer, *COMET: A Recipe for Learning and Using Large Ensembles on Massive Data*,
ICDM 2011), stops when a Gaussian confidence interval around the leading class's vote share
excludes 0.5 (see [GLEE](#glee-gaussian-confidence-interval)).

A fourth one, **margin-based early stopping** (Barbareschi, Emmanuele, *A Margin Based
Early-Stopping Approach for Random Forest Classifiers*, DSN-W 2025), ranks the trees offline
and then lets only the first `N*` of them vote, with `N*` derived from the forest's margin
(see [Margin-based early stopping](#margin-based-early-stopping)). Unlike the other three,
it queries the same number of trees for every sample.

## How to run?

Requires Python 3.12 or newer (the pinned numpy and scipy do not support 3.11).
The pins were last tested on Python 3.15.

### Windows: one script

```bat
setup_windows.bat
```

The script:

1. creates a virtual environment in `.venv` (or reuses an existing one),
2. installs `requirements-dev.txt` (the runtime packages plus Jupyter, matplotlib and seaborn),
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
   pip install -r requirements.txt        # benchmark and dataset scripts only
   pip install -r requirements-dev.txt    # the above, plus notebooks and "# %%" cells
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
| `[[methods]]` | one block per benchmarked method: `type` (`full_voting`, `ib_pruning`, `hyper`, `glee` or `margin`), optional `name` and `enabled`, plus the method's parameters |

`sklearn_soft_voting = true` adds `rf_sklearn` to the benchmark. That is scikit-learn's
`rf.predict`, which averages tree probabilities (soft voting), and the results get the
`agree_rf` and `speedup_vs_rf` columns.
With `false`, the benchmark only runs methods that query individual trees (hard voting).

An `ib_pruning` method takes `pi`, and its table `T{n_trees}_L{l}_pi{pi*100}.npz` is
picked to match each forest. You can give a specific file with `table` instead.
If no table exists for a dataset's number of classes, the method is skipped for that
dataset with a warning.
With `order = "random"` (and an optional `seed`) every sample queries the trees in its own
random permutation instead of the `rf.estimators_` order; the default name becomes
`IB_pi{pi*100}_rand`. Batch mode is noticeably slower in this setting, because samples that
query different trees cannot share one `predict` call.

A `hyper` method takes `pi` and optionally `n_prior` (number of training samples used for
the out-of-bag votes, default 20000) and `smooth` (prior smoothing window, default 5).
It needs no table file.

A `glee` method takes `pi` (the paper's `α` is `1 - pi`) and optionally `min_votes`
(votes required before stopping; default 15, 30 or 45 depending on `α`, as in the paper),
`fpc` (finite population correction, default `true`) and `one_sided` (default `true`).
It needs no table file and no training data, and works for any number of classes.

A `margin` method takes `k` (default 0; larger = more trees) and optionally `ranking`
(`mean_d_m`, `accuracy` or `none`; default `mean_d_m`) and `n_calib` (number of training
samples used for calibration, default 20000). It needs no table file and works for any
number of classes. It has no `pi`: its results are named `MARGIN_k0`, `MARGIN_k-0.5` etc.

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

## GLEE: Gaussian confidence interval

The lazy-evaluation rule of the COMET paper (sec. II-C). Each tree's vote is a Bernoulli
variable, so after `n` votes the leader's vote share `p̂` is roughly Gaussian. Voting stops
when the lower bound of a one-sided `1 - α` confidence interval is above 0.5:

```
p̂ - ρ·δ > 0.5      δ = z_α · sqrt(p̂(1 - p̂) / n)
ρ = sqrt((m - n) / (m - 1))  if n > 0.05·m,  else 1      (finite population correction)
```

with `m` the number of trees and `α = 1 - pi`. No stop is allowed before `min_votes` votes,
because the Gaussian approximation is poor for small `n`. With more than two classes the
leader is treated as class 1 and the runner-up as class 0 (`p̂ = t₁ / (t₁ + t₂)`,
`n = t₁ + t₂`), as the paper suggests.

`Glee.fit` turns the rule into a threshold table `prog[t₂]` = fewest leader votes that stop
(a few milliseconds for 501 trees), so prediction is the same loop and lookup as IB pruning.
The table depends only on `T` and the parameters: no file, no training data, no limit on the
number of classes. `fpc = false` and `one_sided = false` give the paper's G1, G2 and G2-FPC
variants; the defaults are G1-FPC, which the paper calls GLEE.

Only this rule is taken from COMET. Its training side (MapReduce, IVoting forests) builds the
ensemble and is outside the scope of inference on a fitted forest.

## Margin-based early stopping

The method of Barbareschi and Emmanuele (DSN-W 2025). The margin of a sample is
`M = (v_c - v_c*) / N`: votes for the true class minus votes for the best other class, over
the number of trees. It is positive for correctly classified samples.

Offline, `fit(rf, X_train, y_train)`:

1. ranks the trees. The default `mean_d_m` is the MeanD-M criterion the paper uses: a tree's
   score is the mean drop of the margin when that tree is removed from the forest.
   `accuracy` ranks by single-tree accuracy and `none` keeps the forest's order;
2. computes the mean `mu` and standard deviation `sigma` of the margin over the correctly
   classified calibration samples and sets

   ```
   N* = N · (mu + k · sigma)        rounded, kept within [1, N]
   ```

At prediction time every sample is voted on by the first `N*` trees of the ranking, so
`trees_mean = trees_median = trees_max = N*`. `set_k(k)` changes `k` after `fit` without
recalibrating (the paper's on-board tuning of `k`; the tuning policy itself is not specified
in the paper and is not implemented).

The paper calibrates on half of the test set. Here the calibration set is the training data
with out-of-bag votes (each sample is voted on only by trees that did not see it), so the
whole test set is left for measurement, as for the other methods. This needs a forest
trained with `bootstrap=True`.

There is no confidence level: `k` trades trees for agreement directly, and `agree_full` is
not checked against a `pi`. Because `N*` grows with the mean margin, the method keeps *more*
trees on easy datasets (large margins) than on hard ones.

## Running the benchmark

[`src/testing.py`](src/testing.py) reads everything from `config.toml`.
For every dataset in `benchmark.datasets` and every seed in `benchmark.seeds`, it trains a forest
(`[forest]`) and runs the selected methods (`sklearn_soft_voting` and `[[methods]]`) in the
selected `modes`:

- **batch**: the whole test set at once;
- **online**: the first `n_online` samples, predicted one by one.

The default config compares `rf_sklearn`, `full_voting`, `IB_pi98`, `HYPER_pi98`, `GLEE_pi98` and `MARGIN_k0` on the binary synthetic
datasets (501 trees, depth 5, seeds 1–3).

Run it from `src/`:

```sh
cd src
python testing.py
```

You can also run its `# %%` cells interactively in VS Code.
Results are written to `data/<benchmark.out_dir>/`. Each (dataset, seed) run is saved as soon
as it finishes (`runs/<dataset>_seed<seed>.npz`, and the `.csv` tables are rewritten), so an
interrupted benchmark keeps what it has computed. After a run the forest and the data are
released (`benchmark.release()`); only the predictions and timings stay in memory.

Reported per method: accuracy, macro precision/recall/F1, agreement with sklearn and with
full voting (`agree_rf`, `agree_full`), prediction time, speed-up, and trees queried per sample
(`trees_mean`, `trees_median`, `trees_max`, `trees_frac`).
The `*_rf` and `*_full` columns appear only when `rf_sklearn` and `full_voting` are in the benchmark.

### Adding an inference method

The code is type-annotated and keeps docstrings short; the reasoning behind each module is in
[`implementacja.md`](implementacja.md) (in Polish).

Create a module in [`src/methods/`](src/methods/) with a subclass of `InferenceMethod`
(from `methods/base.py`) and implement:

- `predict_batch(X) -> (y_pred, n_used)`
- `predict_one(x) -> (y, n_used)`

After `fit`, `self.trees` is a list of `FastTree` objects (from `methods/base.py`), not
scikit-learn estimators. `tree.predict(X)` returns class indices and skips scikit-learn's
per-call input validation, so call `X = self._X32(X)` once at the start of a prediction
(it converts to `float32` and checks the shape) and pass that `X` to the trees.

An early-stopping method only needs a stopping rule: subclass `SequentialPruning` and
implement `_stop_batch(votes) -> bool mask` (the query loops are shared by IB pruning, HYPER and GLEE).
A method that votes with a fixed list of trees subclasses `FixedVoting` and sets `self.voters`
in `fit` (the voting loops are shared by `full_voting` and the margin method).

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
                                # pass X_train=..., y_train=... for methods that need training
                                # data (hyper: X_train, margin: both)
m.add_method(MyMethod())
```

## Project layout

```
instrukcja.md                what the project and the methods do, data and result formats (Polish)
implementacja.md             code-level notes: table conventions, derivations, non-obvious spots (Polish)
config.toml                  project settings (datasets, benchmark, forest, methods)
setup_windows.bat            venv + requirements + datasets
requirements.txt             runtime packages
requirements-dev.txt         runtime + notebook tooling
data/
  generated/                 synthetic datasets (generated, git-ignored)
  real_data/                 UCI datasets (downloaded, git-ignored)
  tablice/                   IB pruning stop tables
  results/                   benchmark output (git-ignored)
src/
  config.py                  reads config.toml
  methods/                   inference methods, one module each
    base.py                  InferenceMethod interface, FastTree (tree evaluation without
                             scikit-learn's per-call validation), FixedVoting (fixed tree
                             list), SequentialPruning (per-sample early-stopping loop)
    voting.py                rf_sklearn (soft voting), full_voting (hard voting)
    ib_pruning.py            IB pruning (uniform prior, table from file)
    hyper.py                 HYPER (out-of-bag prior, table computed in fit)
    glee.py                  GLEE (Gaussian confidence interval, from the COMET paper)
    margin.py                margin-based early stopping (tree ranking + N* from the margin)
    oob.py                   out-of-bag predictions on the training data (used by hyper, margin)
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
