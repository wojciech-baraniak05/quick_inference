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
