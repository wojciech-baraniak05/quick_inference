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
