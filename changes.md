# Changes: margin-based early stopping (Barbareschi, Emmanuele)

Adds the method from Barbareschi, Emmanuele, *A Margin Based Early-Stopping Approach for
Random Forest Classifiers* (DSN-W 2025, DOI 10.1109/DSN-W65791.2025.00050), removes the code
duplication it would otherwise have created, and fixes one bug found in HYPER on the way.

Note on naming: the request said "hyper method", but HYPER (Soto et al., 2016) was already in
the project and the attached paper describes a different method. That method is what was
added, under the name `margin`.

This file was `chages.md`; it is renamed to `changes.md` (history kept with `git mv`).

## 1. New method: `margin`

New module [src/methods/margin.py](src/methods/margin.py), registered as `type = "margin"` in
[src/methods/registry.py](src/methods/registry.py) and added to [config.toml](config.toml)
with `k = 0.0`.

```
margin of a sample     M  = (v_c - v_c*) / N            (sec. II-B)
trees used at runtime  N* = N · (mu + k · sigma)        (sec. III-A)
```

`v_c` = votes for the true class, `v_c*` = votes for the best other class, `mu` / `sigma` =
mean and standard deviation of `M` over correctly classified calibration samples.

Offline (`fit`): rank the trees, compute `mu`, `sigma` and `N*`.
At prediction time: every sample is voted on by the first `N*` trees of the ranking.

| Piece | What it does |
|---|---|
| `predykcje_kalibracyjne(rf, X_train, y_train, n_calib)` | calibration set: per-tree out-of-bag predictions on the training data (`-1` where a tree saw the sample) and true class indices |
| `margines(votes, y)` | margin of every sample |
| `statystyki_marginesu(votes, y)` | `mu`, `sigma` over correctly classified samples |
| `liczba_drzew(T, mu, sigma, k)` | `N*`, rounded to the nearest integer and kept within `[1, T]` |
| `spadek_marginesu(pred, y, l)` | MeanD-M score of every tree: mean drop of the margin when that tree is removed |
| `trafnosc_drzew(pred, y)` | single-tree accuracy (alternative ranking) |
| `ranking_drzew(pred, y, l, ranking)` | tree order for `"mean_d_m"`, `"accuracy"` or `"none"` |
| `MarginEarlyStop(k, ranking, n_calib, name)` | the method; results are named `MARGIN_k0`, `MARGIN_k-0.5` etc. |
| `MarginEarlyStop.set_k(k)` | changes `k` (and `N*`) after `fit` without recalibrating |

### Implementation choices that go beyond the paper's text

- **Calibration set.** The paper takes `T_off` = half of the test set and evaluates on the
  other half. Here `T_off` is the training data with out-of-bag votes (each sample is voted on
  only by trees that did not see it), so the whole test set stays for measurement, as for the
  other methods. Consequences: the margin is divided by the number of trees that voted for the
  sample (about 37 % of the forest) instead of `N`, and the forest must use `bootstrap=True`.
- **MeanD-M is computed in one pass.** The paper describes it as "for each tree, evaluates the
  average reduction in terms of margin", then ranks. That is what is implemented: one score per
  tree against the full forest. The original MeanD-M paper (Yang et al., 2012) was not
  consulted, so an iterative variant (re-scoring after each removal), if that is what it
  defines, is not implemented.
- **A tree's score is averaged over the samples it votes on** (its out-of-bag samples).
- **Other rankings.** The paper lists similarity-based, accuracy-based and margin-based
  ranking as options. `accuracy` and `none` (forest order) are available next to `mean_d_m`;
  similarity-based ranking is not specified in the paper and is not implemented.
- **"Correctly classified"** means the majority vote (`argmax`, as in `full_voting`) equals
  the true class.
- **Rounding of `N*`** is not given in the paper: nearest integer, kept within `[1, N]`.
- **Tuning `k` after deployment.** The paper says `k` can be raised or lowered on-board
  depending on observed accuracy but gives no algorithm. `set_k` makes the change possible;
  no tuning policy is implemented.
- **`n_calib` (default 20000)** limits the calibration samples, like `n_prior` in HYPER.
- **No `pi`.** The final `agree_full ≥ pi` check in `testing.py` does not cover this method.

### What to expect from it

- It is a static method: the same `N*` trees for every sample. IB pruning, HYPER and GLEE
  decide per sample.
- `N*` grows with the mean margin, so the method keeps more trees on easy datasets than on
  hard ones, as the paper's formula says.

## 2. Interface change: labels reach `fit`

The margin needs true labels, which no method needed before.

- `InferenceMethod.fit(rf, X_train=None, y_train=None)` in [src/methods/base.py](src/methods/base.py);
  `IBPruning`, `Hyper` and `Glee` take the extra argument and ignore it.
- `methods_rf(rf, methods, X_train=None, y_train=None)` in
  [src/benchmarking/forest_methods.py](src/benchmarking/forest_methods.py) passes both to every method.
- [src/testing.py](src/testing.py) passes `y_train`.

Existing calls without `y_train` keep working; `MarginEarlyStop` raises a clear error if the
training data or labels are missing.

## 3. Redundancy removed

| Before | After |
|---|---|
| `FullVoting` had its own `predict_batch` / `predict_one` loops; the margin method needs the same loops over a subset of trees and would have copied both | new `FixedVoting` base class in `base.py` owns both loops and votes with `self.voters`; `FullVoting` is now only a name (all trees), `MarginEarlyStop` only sets `voters` |
| the out-of-bag code (bootstrap indices, random subsample, data checks) lived inside `hyper.py`; the margin method needs the same thing per tree and would have copied it | new module [src/methods/oob.py](src/methods/oob.py): `probka_uczaca`, `predykcje_oob` (generator, one tree at a time) and `glosy_oob`; `hyper.py` imports `glosy_oob` |
| two change logs would have existed (`chages.md` and the requested `changes.md`) | one file, `changes.md` |

`FullVoting` runs the same loop as before (same statements, `self.voters` is the same list as
`self.trees`), so the reference timings do not change.

Left as is on purpose:

- `MarginEarlyStop` is not a `SequentialPruning` subclass. That loop checks a stop rule after
  every tree for every sample, which a fixed `N*` does not need.
- The example class `PierwszeK` in `instrukcja.md` (chapter 10) still implements both predict
  methods itself, because it illustrates the `InferenceMethod` contract.
- `src/__pycache__/decision_tree.cpython-315.pyc` is still tracked in git although `*.pyc` is
  ignored and no `decision_tree` module exists. Not removed; it is unrelated to this change.

## 4. Bug fixed in HYPER (found while testing)

`prior_z_glosow` in [src/methods/hyper.py](src/methods/hyper.py) smooths the prior with
`scipy.ndimage.uniform_filter`, which can leave values around `-1e-20` where the result should
be 0. `tablica_hyper` then took the logarithm of a negative number, the whole threshold table
became `nan`, and HYPER silently stopped only when the remaining trees could no longer change
the winner. The prior is now clipped at 0 after smoothing (one line).

| forest | HYPER_pi98 before | after |
|---|---|---|
| 501 trees, 2 classes | 20.2 trees, agree_full 0.9815 | identical |
| 501 trees, 3 classes | 352.9 trees, agree_full 1.0000 | 47.6 trees, agree_full 0.9785 |
| 101 trees, 3 classes | 75.5 trees, agree_full 1.0000 | 31.4 trees, agree_full 0.9930 |
| 60 trees, 4 classes | 44.9 trees, agree_full 1.0000 | 21.7 trees, agree_full 0.9940 |

(`make_classification` data, depth 5.) No two-class case tested was affected, so the binary
results reported below for HYPER in earlier sections still hold. Any earlier HYPER result with
3 or more classes should be re-run.

## 5. Documentation

- [README.md](README.md): intro, config keys, new section "Margin-based early stopping",
  default method list, "adding a method" (`FixedVoting`, `y_train`), project layout.
- [instrukcja.md](instrukcja.md): new chapter 2c, method table, flow diagram, config example,
  notes on tree order, `FixedVoting`, the out-of-bag module and the `fit` signature.

## 6. Verification

Run with temporary scripts (no test files were added to the repository).

- **MeanD-M and accuracy scores = brute force.** For random predictions with missing votes
  (2, 3 and 5 classes), every tree's score matches removing the tree and recomputing the margin.
- **Out-of-bag refactor.** `glosy_oob` returns the same votes as the version in the previous
  commit (2 and 4 classes, with and without subsampling); the calibration predictions sum to
  the same votes; non-consecutive class labels are mapped to class indices correctly.
- **`FullVoting`** gives the same predictions as a direct vote count, with `n_used = T`.
- **`MarginEarlyStop`**, all three rankings, 2 and 4 classes: batch and online give identical
  predictions; `n_used = N*` for every sample; predictions equal a direct vote of the first
  `N*` ranked trees; `N*` is monotone in `k`; a very large `k` reproduces `full_voting`.
- **`testing.py` end to end** with the default method list, `binary_small` and
  `hard_nonlinear`, seeds 1–2 (datasets and output folder overridden in memory, config file
  unchanged). Mean over the four runs:

  | method | agree_full | trees_mean | speedup_vs_full batch | speedup_vs_full online |
  |---|---|---|---|---|
  | `IB_pi98` | 0.9843 | 29.7 | 2.20 | 16.5 |
  | `HYPER_pi98` | 0.9780 | 26.8 | 2.14 | 17.0 |
  | `GLEE_pi98` | 0.9920 | 37.6 | 2.14 | 13.2 |
  | `MARGIN_k0` | 0.9832 | 332.3 | 1.49 | 1.53 |

  At a similar `agree_full` the margin method queries about 11 times more trees than IB
  pruning, because it cannot stop early on easy samples.
- **Effect of `k`** (501 trees, depth 5, seeds 1–2, batch, at most 6000 test samples).
  Each cell is `N*` / agree_full / accuracy:

  | dataset (classes) | full_voting accuracy | k = -1.5 | k = -1 | k = -0.5 | k = 0 | k = 0.5 |
  |---|---|---|---|---|---|---|
  | `binary_small` (2) | 0.8633 | 155 / 0.983 / 0.860 | 221 / 0.983 / 0.860 | 286 / 0.992 / 0.862 | 352 / 0.995 / 0.862 | 418 / 0.997 / 0.863 |
  | `hard_nonlinear` (2) | 0.6871 | 112 / 0.942 / 0.703 | 179 / 0.952 / 0.698 | 246 / 0.965 / 0.692 | 313 / 0.971 / 0.689 | 380 / 0.978 / 0.686 |
  | `noisy_labels` (2) | 0.8028 | 138 / 0.979 / 0.802 | 206 / 0.984 / 0.802 | 274 / 0.987 / 0.802 | 342 / 0.992 / 0.802 | 409 / 0.994 / 0.802 |
  | `multi_small` (4) | 0.7233 | 58 / 0.937 / 0.737 | 121 / 0.948 / 0.733 | 184 / 0.962 / 0.728 | 247 / 0.977 / 0.725 | 310 / 0.983 / 0.730 |

  More trees and higher agreement with growing `k`, as in the paper's Fig. 4 and 5. Accuracy
  stays within about 1.6 percentage points of the full forest and is sometimes above it.
- **Effect of the ranking** at `k = -1` (same runs), agree_full / accuracy:

  | dataset | `mean_d_m` | `accuracy` | `none` |
  |---|---|---|---|
  | `binary_small` | 0.983 / 0.860 | 0.983 / 0.860 | 0.992 / 0.858 |
  | `hard_nonlinear` | 0.952 / 0.698 | 0.954 / 0.698 | 0.977 / 0.692 |
  | `noisy_labels` | 0.984 / 0.802 | 0.984 / 0.802 | 0.989 / 0.802 |
  | `multi_small` | 0.948 / 0.733 | 0.953 / 0.718 | 0.965 / 0.708 |

  Ranking makes the sub-forest agree less with the full vote than the same number of unranked
  trees, and on two of the four datasets it is more accurate. These are two seeds per dataset,
  so differences of a few thousandths are within noise.
- `MarginEarlyStop.fit` takes 0.2–0.8 s for 501 trees.

Not run: the full default benchmark through `testing.py` (all 11 binary datasets, 3 seeds),
and the paper's own datasets (Avila, AI4I, Dry Bean), which are not in the project. The
paper's figure of about 53 of 100 trees for a 2 % accuracy loss was therefore not reproduced.

---

# Changes: GLEE method (COMET paper)

Adds the lazy ensemble evaluation rule from Basilico, Munson, Kolda, Dixon, Kegelmeyer,
*COMET: A Recipe for Learning and Using Large Ensembles on Massive Data* (ICDM 2011,
arXiv:1103.2068), sec. II-C. The paper calls it GLEE (Gaussian Lazy Ensemble Evaluation).

## What was added

New module [src/methods/glee.py](src/methods/glee.py), registered as `type = "glee"` in
[src/methods/registry.py](src/methods/registry.py) and added to [config.toml](config.toml)
with `pi = 0.98` next to `IB_pi98` and `HYPER_pi98`.

```
stop when  p̂ - ρ·δ > 0.5
δ = z_α · sqrt(p̂(1 - p̂) / n)
ρ = sqrt((m - n) / (m - 1))  if n > 0.05·m,  else 1
```

| Piece | What it does |
|---|---|
| `min_glosow(alfa)` | the paper's minimum number of votes before a stop: 15 / 30 / 45 for `α` = 1e-2 / 1e-3 / 1e-4 |
| `tablica_glee(T, alfa, min_votes, fpc, one_sided)` | threshold table `prog[t_2]` = fewest leader votes `t_1` at which the rule stops |
| `czy_stop_glee(votes, prog)` | vectorised stop check (two largest vote counts, one lookup) |
| `Glee(pi, min_votes, fpc, one_sided, name)` | the method, a `SequentialPruning` subclass; results are named `GLEE_pi98` etc. |

No interface changes: `Glee` only supplies the stopping rule to the existing loops.

## Implementation choices that go beyond the paper's text

- **Only the inference rule is implemented.** COMET's training side (MapReduce, IVoting
  forests) builds the ensemble; this project benchmarks inference on a fitted scikit-learn forest.
- **`pi` instead of `α`.** The constructor takes `pi = 1 - α` so that configs, method names and
  the `agree_full < pi` check in `testing.py` work the same as for IB pruning and HYPER.
- **Multiclass.** The paper only says the leader is treated as class 1 and the runner-up as
  class 0. Here `p̂ = t_1 / (t_1 + t_2)` and `n = t_1 + t_2`, with `m` still the full forest
  size. Votes for other classes are ignored, which makes the correction `ρ` slightly
  conservative (it treats them as trees not yet queried). For two classes this is exactly
  the paper's rule.
- **`min_votes`** counts `n = t_1 + t_2`. For `α` between the paper's three values the
  stricter (larger) minimum is used.
- **Table instead of evaluating the formula per vote**, as the paper also does. For a fixed
  `t_2` the lower bound grows with `t_1`, so one threshold per `t_2` is exact.
- **Tree order.** The paper draws trees at random without replacement. As for IB pruning and
  HYPER, trees are queried in the forest's own order, which is exchangeable for a random forest.
- **No "result already certain" shortcut.** The rule is the paper's formula only; it does not
  additionally stop when the remaining trees can no longer change the winner.
- `fpc = false` and `one_sided = false` expose the paper's G1, G2 and G2-FPC variants;
  the defaults are G1-FPC.

## Verification

Run with a temporary script (no test files were added to the repository):

- **Table = formula.** For `T` in {51, 100, 501}, `α` in {0.05, 0.02, 0.01, 0.001} and all
  four `fpc` / `one_sided` combinations, the table gives the same decision as a scalar
  evaluation of the formula in every vote state, and the rule is monotone in `t_1`.
- **Paper's simulation (sec. III)**: vote share `p ~ U[0, 1]`, Bernoulli votes.

  | m | α | rule | fraction of ensemble | relative error | disagreement with full vote |
  |---|---|---|---|---|---|
  | 501 | 0.02 | GLEE (G1-FPC) | 0.103 | 0.0074 | 0.0125 |
  | 501 | 0.02 | G1 | 0.128 | 0.0064 | 0.0095 |
  | 501 | 0.02 | G2-FPC | 0.120 | 0.0044 | 0.0076 |
  | 501 | 0.02 | IB pruning (`T501_L2_pi98`) | 0.092 | 0.0115 | 0.0167 |
  | 1000 | 0.01 | GLEE (G1-FPC) | 0.081 | 0.0051 | 0.0077 |
  | 1000 | 0.001 | GLEE (G1-FPC) | 0.130 | 0.0005 | 0.0008 |

  Relative error and disagreement stay below `α`, as in the paper's Fig. 1c.
- **Batch and online** give identical predictions and tree counts (2, 4 and 20 classes).
- **Forests** (501 trees, depth 5, seeds 1–2, batch, `pi = 0.98`):

  | dataset | classes | `GLEE_pi98` agree_full / trees | `IB_pi98` | `HYPER_pi98` |
  |---|---|---|---|---|
  | `binary_small` | 2 | 0.993–1.000 / 34–35 | 0.983–0.993 / 24–30 | 0.980–0.983 / 23–25 |
  | `hard_nonlinear` | 2 | 0.986–0.989 / 39–43 | 0.978–0.982 / 30–36 | 0.967–0.981 / 26–34 |
  | `multi_small` | 4 | 0.990–0.997 / 63–67 | no table | too many states |
  | `multi_many_classes` | 20 | 0.989–0.990 / 210 | no table | too many states |

  GLEE never dropped below `pi` here but queries more trees than IB pruning and HYPER at
  the same `pi`: it cannot stop before 15 votes, while IB stops after 5 unanimous ones.
- `Glee.fit` takes about 3 ms for 501 trees.

Not run: the full default benchmark through `testing.py` (all 11 binary datasets, 3 seeds).

---

# Changes: HYPER method

Adds the dynamic ensemble pruning method from Soto, Suárez, Martínez-Muñoz,
*An urn model for majority voting in classification ensembles* (NIPS 2016),
and removes the code duplication that adding it would otherwise have created.

## 1. New method: `hyper`

New module [src/methods/hyper.py](src/methods/hyper.py).

The paper models the forest, for one sample, as an urn of `T` marbles (trees) in `l` colours
(classes) with unknown composition `T = (T_1..T_l)`. Querying trees is drawing without
replacement, so after votes `t`:

```
P(T | t)  ∝  C(T_1, t_1) · … · C(T_l, t_l) · P(T)             (eq. 2)
P*(t, T)  =  sum of P(T | t) over compositions with the same winner as t   (eq. 3)
stop when P*(t, T) ≥ pi
```

With a uniform `P(T)` this is the existing IB pruning (SIBA). HYPER estimates `P(T)` from data.

| Piece | What it does |
|---|---|
| `glosy_oob(rf, X_train, n_prior)` | out-of-bag votes of the training samples (each sample is voted on only by trees that did not see it) |
| `prior_z_glosow(votes, T, smooth)` | non-parametric prior (sec. 2.2): votes rescaled to `T` trees, histogram of compositions, sliding-window smoothing of width 5 |
| `tablica_hyper(prior, pi)` | threshold table, one per leading class: `prog[c, votes of the other classes]` = fewest votes for `c` at which `P* ≥ pi` |
| `czy_stop_hyper(votes, prog)` | vectorised stop check |
| `Hyper(pi, n_prior, smooth, name)` | the method; `fit(rf, X_train)` builds prior and table, results are named `HYPER_pi98` etc. |
| `Hyper.expected_trees()` | the paper's Monte Carlo estimate of the mean number of queried trees, from the prior alone |

Registered as `type = "hyper"` in [src/methods/registry.py](src/methods/registry.py) and
added to [config.toml](config.toml) with `pi = 0.98` next to `IB_pi98`.

### Implementation choices that go beyond the paper's text

- **Table computation.** Instead of summing eq. 3 over all compositions for every vote state,
  the table comes from a backward recursion `f(t) = Σ_i f(t + e_i)` over vote counts, done
  layer by layer in NumPy. Cost: about 0.03 s for `T=501, l=2` and 7 s for `T=501, l=3`.
- **Table is not stored.** It depends on the data, so it is computed in `fit` for every forest.
- **Size limit.** More than 2,000,000 prior cells `(T+1)^(l-1)` raises `ValueError`, which
  `testing.py` already turns into "skip this method for this dataset". With 501 trees this
  means 2 or 3 classes.
- **`n_prior` (default 20000).** Out-of-bag votes are collected on a random subset of the
  training samples to keep `fit` cheap on large datasets. `n_prior` larger than the training
  set uses all samples.
- **Non-monotone `P*`.** The threshold is placed after the last state with `P* < pi`, so every
  stop satisfies `P* ≥ pi` even if the smoothed prior makes `P*` non-monotone.
- **Impossible states.** Vote states with zero probability under the prior never stop early,
  except when the remaining votes cannot change the winner.
- **Ties** in the full vote are resolved as `full_voting` resolves them (`argmax`, lower class
  index), not counted as a loss. IB pruning's tables count a tie as a loss.
- **Stop condition** is `P* ≥ pi` (the paper writes `>`), to match IB pruning.
- **Multiclass rounding.** Rescaled votes are rounded with the largest-remainder rule so they
  sum to exactly `T`.
- Bootstrap indices come from scikit-learn's private `rf._get_estimators_indices()` (lazy,
  low memory), falling back to the public `rf.estimators_samples_`.

## 2. Interface change: training data reaches `fit`

HYPER needs the forest's training data, which no method needed before.

- `InferenceMethod.fit(rf, X_train=None)` in [src/methods/base.py](src/methods/base.py).
- `methods_rf(rf, methods, X_train=None)` in
  [src/benchmarking/forest_methods.py](src/benchmarking/forest_methods.py) passes it to every method.
- [src/testing.py](src/testing.py) passes `X_train`.

Existing calls without `X_train` keep working; `Hyper` raises a clear error if it is missing.

## 3. Redundancy removed

| Before | After |
|---|---|
| `IBPruning` had its own `predict_batch` / `predict_one` loops; HYPER would have copied both | new `SequentialPruning` base class in `base.py` owns both loops; `IBPruning` and `Hyper` only supply `_stop_batch` / `_stop_one` |
| the `pi → "98"` formatting was written twice (table file name, method name) | one helper `pi_tag(pi)` in [src/generate_table_IB.py](src/generate_table_IB.py), also used for `HYPER_pi…` |
| three table loaders in `ib_pruning.py` (`_wczytaj_npz`, `wczytaj_plik_tablicy`, `tablica_dla`) with two copies of the "file missing" check | `_wczytaj_npz` + one `wczytaj_tablice(plik, podpowiedz)` |
| `IBPruning.from_npz(path)` duplicated `IBPruning(table=path)` and was unused | removed |
| `methods_rf.run` re-implemented the mode check | calls `check_modes` |
| `testing.py` rebuilt the reference-method names from `SklearnRF` / `FullVoting` | uses `benchmark.REF_RF` / `benchmark.REF_FULL` |

Left as is on purpose: `FullVoting` keeps its own loop. Routing it through `SequentialPruning`
would add per-tree indexing to the reference method and change its timings, which
`speedup_vs_full` depends on.

## 4. Documentation

- [README.md](README.md): HYPER section, config keys, project layout, note on `SequentialPruning`.
- [instrukcja.md](instrukcja.md): new chapter 2a on HYPER, updated flow diagram, config example
  and "adding a method" notes.

## 5. Verification

Run with temporary scripts (no test files were added to the repository):

- **Uniform prior = IB pruning.** `tablica_hyper` with a uniform prior gives the same stop
  decision as the stored IB tables in every state for `T51_L2_pi95` (1,325 states) and
  `T501_L2_pi98` (125,750 states). For `T501_L3_pi98` 1,647 of 576,000 sampled states differ,
  all of them states where HYPER stops and IB does not; they are borderline states where the
  tie convention above moves `P*` across 0.98 (checked: 0.9787 with ties as losses, 0.9800
  with `argmax` ties).
- **Brute force.** For `T=12, l=3` and a random sparse prior, all 363 vote states match a
  direct evaluation of eq. 3.
- **Out-of-bag votes** match a direct computation from `rf.estimators_samples_`.
- **Batch and online** give identical predictions and tree counts.
- **Benchmark** (`testing.py`, default config, `binary_small` and `hard_nonlinear`, 3 seeds, batch):

  | method | agree_full | trees_mean |
  |---|---|---|
  | `IB_pi98` | 0.9867 | 32.2 |
  | `HYPER_pi98` | 0.9817 | 29.2 |

  HYPER queries fewer trees and its agreement sits closer to `pi`, as in the paper. It is
  not a hard floor: `hard_nonlinear` seed 2 gave 0.967 for HYPER (IB: 0.978), so the final
  check in `testing.py` reports HYPER runs below `pi` more often than IB runs.

Not run: the full default benchmark over all 11 binary datasets, and any dataset with 3 classes
through `testing.py` (the 3-class path was exercised on a synthetic `make_classification` set
with 101 trees).
