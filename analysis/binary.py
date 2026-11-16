# %%
"""Analiza wyników benchmarku dla klasyfikacji binarnej: jakość, koszt, zgodność z pełnym
lasem (hard voting), szybkość i testy statystyczne równoważności metod z full_voting.

Czyta plik wyników w nowym formacie (jak data/results/classification_all.parquet, tylko
dane surowe: y_true, y_pred, n_used, time_s). Domyślnie pracuje na małym pliku testowym
(binary_test.parquet, tworzony z pełnego pliku w komórce 1); pełna analiza po ustawieniu
RESULTS_FILE = FULL_FILE.

Uruchomienie: komórkami (# %%) w VS Code albo całość z katalogu głównego:
    python analysis/binary.py
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

try:
    from IPython.display import display
except ImportError:                                 # bez IPython: zwykły wydruk
    def display(obj: object) -> None:
        print(obj.to_string() if isinstance(obj, pd.DataFrame) else obj)


def _find_root() -> Path:
    """Katalog główny repozytorium (ten z src/analysis): od pliku, a w Interactive Window,
    gdzie __file__ bywa nieustawione, od katalogu roboczego w górę."""
    starts = [Path(__file__).resolve().parent] if "__file__" in globals() else []
    starts.append(Path.cwd().resolve())
    for start in starts:
        for p in (start, *start.parents):
            if (p / "src" / "analysis" / "metrics.py").exists():
                return p
    raise FileNotFoundError("repository root (directory with src/analysis) not found")


ROOT = _find_root()
# src na początek ścieżki: "import analysis" ma trafić w src/analysis, nie w ten katalog
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from analysis.metrics import REF_FULL, REF_RF, compute_metrics  # noqa: E402

FULL_FILE = ROOT / "data" / "results" / "classification_all.parquet"
TEST_FILE = ROOT / "data" / "results" / "binary_test.parquet"
RESULTS_FILE = TEST_FILE            # FULL_FILE: pełna analiza (wszystkie seedy)
TEST_SEEDS = [1, 2, 3]              # seedy w pliku testowym
REBUILD_TEST = False                # True: plik testowy od nowa (np. po nowym przebiegu)

REF = REF_FULL                      # "full_voting": pełny las, hard voting
ALPHA = 0.05
EQ_MARGIN = 0.005                   # margines równoważności TOST: ±0.5 pp
QUALITY_MODE = "batch"              # jakość, zgodność i testy na pełnym X_test

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 300)
pd.set_option("display.float_format", "{:.4f}".format)
sns.set_theme(style="whitegrid", context="notebook")
BLUE, ORANGE = sns.color_palette("colorblind")[:2]


def show(fig: plt.Figure) -> None:
    """Pokazuje wykres i zamyka figurę (przy uruchomieniu całości nie zbierają się)."""
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def title(text: str) -> None:
    print(f"\n=== {text} ===")


# %% [markdown]
# ## Pomocnicze: metryki z macierzy pomyłek, statystyki, testy

# %%
def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Macierz pomyłek 2x2 (wiersze: klasa prawdziwa, kolumny: przewidziana)."""
    t = np.searchsorted(labels, y_true)
    p = np.searchsorted(labels, y_pred)
    return np.bincount(t * len(labels) + p, minlength=len(labels) ** 2).reshape(
        len(labels), len(labels))


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def quality(C: np.ndarray, minority: int) -> dict[str, float]:
    """Metryki jakości z macierzy pomyłek; minority = indeks klasy mniejszościowej.

    Balans klas: w binarnej klasyfikacji macro recall to balanced accuracy, więc widok
    „każda klasa waży tyle samo” dają bal_accuracy i f1_macro, a precision / recall / f1 są
    liczone dla klasy mniejszościowej (klasa „pozytywna”, ta, na której błędy kosztują
    najwięcej). MCC korzysta z całej macierzy i nie premiuje zgadywania klasy większościowej."""
    n = C.sum()
    tp = np.diag(C)
    recall = np.array([_div(tp[c], C[c].sum()) for c in range(len(C))])
    precision = np.array([_div(tp[c], C[:, c].sum()) for c in range(len(C))])
    f1 = np.array([_div(2 * p * r, p + r) for p, r in zip(precision, recall)])
    TP, FN = C[minority, minority], C[minority].sum() - C[minority, minority]
    FP = C[:, minority].sum() - TP
    TN = n - TP - FN - FP
    denom = np.sqrt(float(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return {
        "accuracy": tp.sum() / n,
        "bal_accuracy": recall.mean(),
        "precision": precision[minority],
        "recall": recall[minority],
        "f1": f1[minority],
        "f1_macro": f1.mean(),
        "mcc": _div(float(TP) * TN - float(FP) * FN, denom),
    }


def holm(p: np.ndarray) -> np.ndarray:
    """Poprawka Holma (FWER) dla wektora p-wartości; NaN zostają NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    order = np.argsort(q)
    adj = np.minimum(np.maximum.accumulate((len(q) - np.arange(len(q))) * q[order]), 1.0)
    res = np.empty_like(q)
    res[order] = adj
    out[ok] = res
    return out


def tost(d: np.ndarray, eps: float, alpha: float = ALPHA) -> dict[str, float | str]:
    """Test równoważności TOST dla różnic sparowanych d (średnia w ±eps) + dwustronny t-test
    różnicy. Werdykt: equivalent (TOST p < alpha), better / worse (istotna różnica, bez
    równoważności), inconclusive (żaden test nie rozstrzyga)."""
    d = np.asarray(d, dtype=float)
    n, mean = len(d), d.mean()
    if n < 2:
        return {"n": n, "mean": mean, "ci_lo": np.nan, "ci_hi": np.nan, "p_tost": np.nan,
                "p_diff": np.nan, "verdict": "n<2"}
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:                                     # wszystkie różnice równe
        p_tost = 0.0 if abs(mean) < eps else 1.0
        p_diff = 1.0 if mean == 0 else 0.0
        lo = hi = mean
    else:
        df = n - 1
        p_tost = max(stats.t.sf((mean + eps) / se, df), stats.t.cdf((mean - eps) / se, df))
        p_diff = 2 * stats.t.sf(abs(mean) / se, df)
        h = stats.t.ppf(1 - alpha, df) * se         # 90% CI: dualny do TOST na poziomie 5%
        lo, hi = mean - h, mean + h
    if p_tost < alpha:
        verdict = "equivalent"
    elif p_diff < alpha:
        verdict = "better" if mean > 0 else "worse"
    else:
        verdict = "inconclusive"
    return {"n": n, "mean": mean, "ci_lo": lo, "ci_hi": hi, "p_tost": p_tost,
            "p_diff": p_diff, "verdict": verdict}


def med_std(df: pd.DataFrame, value: str, index: str | list[str], columns: str,
            scale: float = 1.0, digits: int = 4) -> pd.DataFrame:
    """Tabela „mediana ± std” (po seedach albo przebiegach) wartości value."""
    g = df.groupby([*np.atleast_1d(index), columns], sort=False)[value]
    agg = pd.DataFrame({"med": g.median() * scale, "std": g.std() * scale})
    cell = agg.apply(lambda r: f"{r['med']:.{digits}f} ± {r['std']:.{digits}f}", axis=1)
    return cell.unstack(columns)


def critical_difference(k: int, n: int, alpha: float = ALPHA) -> float:
    """Krytyczna różnica średnich rang w teście Nemenyi (Demšar 2006)."""
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6 * n))


def cd_diagram(ranks: pd.Series, cd: float, label: str) -> None:
    """Diagram różnic krytycznych: średnie rangi (1 = najlepsza), kreski łączą metody,
    których rangi różnią się o mniej niż CD (nierozróżnialne testem Nemenyi)."""
    r = ranks.sort_values()
    k = len(r)
    half = (k + 1) // 2
    fig, ax = plt.subplots(figsize=(10, 1.4 + 0.42 * half))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-half - 0.6, 2.2)
    ax.axhline(0, color="0.3", lw=1)
    for x in range(1, k + 1):
        ax.plot([x, x], [0, 0.15], color="0.3", lw=1)
        ax.text(x, 0.35, str(x), ha="center", fontsize=9)
    ax.plot([1, 1 + cd], [1.4, 1.4], color="0.1", lw=2)
    ax.text(1 + cd / 2, 1.65, f"CD = {cd:.2f}", ha="center", fontsize=9)
    for i, (name, rank) in enumerate(r.items()):
        y = -(i % half) - 1.2
        left = i < half
        x_text = 0.55 if left else k + 0.45
        ax.plot([rank, rank, x_text], [0, y, y], color="0.45", lw=0.8)
        ax.text(x_text, y + 0.12, f"{name} ({rank:.2f})", fontsize=8.5,
                ha="left" if left else "right", va="bottom")
    # grupy: maksymalne ciągi metod (wg rangi) o rozpiętości < CD
    vals = r.to_numpy()
    groups, last_end = [], -1
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and j > last_end:
            groups.append((i, j))
            last_end = j
    for g, (i, j) in enumerate(groups):
        y = -0.35 - 0.18 * g
        ax.plot([vals[i] - 0.03, vals[j] + 0.03], [y, y], color="0.1", lw=3)
    ax.set_title(f"{label}: mean ranks (Nemenyi, alpha={ALPHA})", fontsize=10)
    ax.axis("off")
    show(fig)


# %% [markdown]
# ## 1. Plik testowy do szybkich analiz

# %%
def make_test_file(src: Path, dst: Path, seeds: list[int]) -> Path:
    """Plik testowy w tym samym formacie co src: tylko zbiory binarne (l == 2) i podane
    seedy, wiersze i tablice bez zmian. Zapis przez plik tymczasowy i podmianę."""
    raw = pd.read_parquet(src, filters=[("l", "==", 2), ("seed", "in", list(seeds))])
    if raw.empty:
        raise ValueError(f"make_test_file: no binary runs with seeds {seeds} in {src}")
    missing = sorted(set(seeds) - set(raw["seed"]))
    if missing:
        print(f"warning: seeds {missing} not in {src.name}")
    tmp = dst.with_name(dst.name + ".tmp")
    raw.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(dst)
    return dst


if RESULTS_FILE == TEST_FILE and (REBUILD_TEST or not TEST_FILE.exists()):
    make_test_file(FULL_FILE, TEST_FILE, TEST_SEEDS)
    print(f"test file written: {TEST_FILE} ({TEST_FILE.stat().st_size / 2**20:.1f} MB, "
          f"from {FULL_FILE.name}, seeds {TEST_SEEDS})")
print(f"results: {RESULTS_FILE}")

# %% [markdown]
# ## 2. Wczytanie, metryki z src/analysis, kompletność przebiegów

# %%
raw = pd.read_parquet(RESULTS_FILE)
raw = raw[raw["l"] == 2].reset_index(drop=True)
KEY = ["dataset", "seed"]                           # przebieg
dup = raw.duplicated([*KEY, "mode", "method"])
if dup.any():
    raise ValueError("the same (dataset, seed, mode, method) more than once: the file mixes "
                     "runs with different settings; filter them first")

# przebieg (zbiór, seed, tryb) musi mieć wszystkie metody, inaczej porównania sparowane
# (każda metoda na tych samych próbkach) nie mają sensu
METHODS = list(dict.fromkeys(raw["method"]))        # kolejność z pliku
full_set = set(METHODS)
complete = raw.groupby([*KEY, "mode"])["method"].transform(lambda s: set(s) == full_set)
if not complete.all():
    dropped = raw.loc[~complete, [*KEY, "mode"]].drop_duplicates()
    print(f"dropped {len(dropped)} incomplete (dataset, seed, mode) groups:")
    print(dropped.to_string(index=False))
    raw = raw[complete].reset_index(drop=True)
if REF not in METHODS:
    raise ValueError(f"{REF} not in the results: no reference for agreement and tests")

m = compute_metrics(raw)                            # src/analysis: ten sam indeks co raw
T = int(raw["T"].iloc[0])
MODES = list(dict.fromkeys(raw["mode"]))
EARLY = [x for x in METHODS if x not in (REF, REF_RF)]   # metody wczesnego zatrzymania
HAS_RF = REF_RF in METHODS

print(f"{len(raw)} rows | datasets: {raw['dataset'].nunique()} | seeds: "
      f"{sorted(raw['seed'].unique().tolist())} | modes: {MODES} | T = {T}")
print(f"methods ({len(METHODS)}): {', '.join(METHODS)}")

# %% [markdown]
# ## 3. Sanity checks

# %%
# y_true to samo dla wszystkich metod w przebiegu
same_y = raw.groupby([*KEY, "mode"])["y_true"].apply(
    lambda s: all(np.array_equal(s.iloc[0], y) for y in s))
print(f"y_true identical within every run: {bool(same_y.all())}")

# tryb online liczy pierwsze n_online próbek X_test: dla metod deterministycznych predykcje
# muszą być prefiksem predykcji batch (losowa kolejność drzew może je zmieniać)
if {"batch", "online"} <= set(MODES):
    b = raw[raw["mode"] == "batch"].set_index([*KEY, "method"])["y_pred"]
    o = raw[raw["mode"] == "online"].set_index([*KEY, "method"])["y_pred"]
    prefix = pd.Series([np.array_equal(o[i], b[i][:len(o[i])]) for i in o.index], index=o.index)
    title("share of runs with online predictions == batch prefix")
    display(prefix.groupby(level="method", sort=False).mean().rename("share").to_frame().T)

# pary metod z identycznymi predykcjami we wszystkich przebiegach (np. *_pair dla 2 klas):
# w testach rangowych liczy się tylko pierwsza z pary (remis z samą sobą zaniża rangi)
bq = raw[raw["mode"] == QUALITY_MODE].set_index([*KEY, "method"])["y_pred"]
runs_q = bq.index.droplevel("method").unique()
pairs = []
for a, c in combinations(METHODS, 2):
    share = np.mean([np.array_equal(bq[(*r, a)], bq[(*r, c)]) for r in runs_q])
    if share >= 0.5:
        pairs.append({"method": a, "same_as": c, "identical_runs": share})
pairs = pd.DataFrame(pairs, columns=["method", "same_as", "identical_runs"])
DUPLICATES = sorted(set(pairs.loc[pairs["identical_runs"] == 1, "same_as"]))
title(f"method pairs with identical predictions in >= 50% of runs ({QUALITY_MODE})")
display(pairs)
print(f"excluded from rank tests (duplicates): {DUPLICATES or 'none'}")

# %% [markdown]
# ## 4. Balans klas i metryki z próbek

# %%
# klasa mniejszościowa zbioru: najrzadsza w y_true po wszystkich seedach
first = raw[(raw["mode"] == QUALITY_MODE) & (raw["method"] == REF)]
balance = []
for ds, g in first.groupby("dataset", sort=False):
    y = np.concatenate(g["y_true"].to_list())
    labels, counts = np.unique(y, return_counts=True)
    balance.append({"dataset": ds, "labels": labels, "minority": labels[counts.argmin()],
                    "n_test": int(g["n_test"].iloc[0]),
                    "minority_share": counts.min() / counts.sum(),
                    "imbalance_ratio": counts.max() / counts.min()})
balance = pd.DataFrame(balance).set_index("dataset").sort_values("imbalance_ratio")
DATASETS = balance.index.tolist()                   # kolejność: od zbalansowanych
title("class balance of the test sets (pooled over seeds)")
display(balance.drop(columns="labels"))


def sample_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Metryki liczone z tablic, wiersz na (zbiór, seed, tryb, metoda): jakość, zgodność
    z full_voting (też w podziale na klasy), anatomia niezgodności i koszt per klasa."""
    rows = []
    for (ds, seed, mode), g in raw.groupby([*KEY, "mode"], sort=False):
        labels = balance.at[ds, "labels"]
        k_min = int(np.searchsorted(labels, balance.at[ds, "minority"]))
        y_true = g["y_true"].iloc[0]
        preds = dict(zip(g["method"], g["y_pred"]))
        used = dict(zip(g["method"], g["n_used"]))
        y_ref = preds[REF]
        ref_ok = y_ref == y_true
        is_min = y_true == labels[k_min]
        for name in g["method"]:
            y = preds[name]
            agree = y == y_ref
            ok = y == y_true
            row = {"dataset": ds, "seed": seed, "mode": mode, "method": name,
                   **quality(confusion(y_true, y, labels), k_min)}
            # zgodność zbalansowana: średnia ze zgodności w każdej klasie prawdziwej
            # (i osobno w klasach przewidzianych przez pełny las), plus na klasie rzadkiej
            by_true = [agree[y_true == c].mean() for c in labels if (y_true == c).any()]
            by_ref = [agree[y_ref == c].mean() for c in labels if (y_ref == c).any()]
            row.update(
                n=len(y), n_agree=int(agree.sum()),
                agree_full=agree.mean(), agree_full_bal=np.mean(by_true),
                agree_full_bal_ref=np.mean(by_ref),
                agree_full_min=agree[is_min].mean() if is_min.any() else np.nan,
                # niezgodności: rację ma metoda (b) albo pełny las (c); McNemar na (b, c)
                n_dis=int((~agree).sum()), b=int((ok & ~ref_ok).sum()),
                c=int((~ok & ref_ok).sum()),
                dis_min=int((~agree & is_min).sum()),
                trees_min=used[name][is_min].mean() if is_min.any() else np.nan,
                trees_maj=used[name][~is_min].mean(),
            )
            rows.append(row)
    return pd.DataFrame(rows)


s = sample_metrics(raw)
# do metryk z próbek dochodzą czas, przyspieszenie i liczby drzew z compute_metrics
cols = ["time_s", "ms_per_sample", "speedup_vs_full", "speedup_vs_rf", "trees_mean",
        "trees_median", "trees_max", "trees_frac", "agree_rf", "pi"]
cols = [c for c in cols if c in m.columns]
df = s.merge(m[[*KEY, "mode", "method", *cols]], on=[*KEY, "mode", "method"])
df["us_per_tree"] = df["time_s"] / (df["trees_mean"] * df["n"]) * 1e6
df["saved_trees"] = 1 - df["trees_frac"]
assert np.allclose(df["agree_full"], df.merge(m, on=[*KEY, "mode", "method"])["agree_full_y"])
q = df[df["mode"] == QUALITY_MODE]

# różnice względem full_voting w tym samym przebiegu (sparowane)
ref_q = q[q["method"] == REF].set_index(KEY)
QUAL = ["accuracy", "bal_accuracy", "precision", "recall", "f1", "f1_macro", "mcc"]
for c in QUAL:
    q = q.assign(**{f"d_{c}": q[c].to_numpy() - ref_q.loc[
        pd.MultiIndex.from_frame(q[KEY]), c].to_numpy()})

# %% [markdown]
# ## 5. Metryki jakości (mediana ± std po seedach)

# %%
title(f"quality per method, median ± std over all runs ({QUALITY_MODE})")
display(pd.concat({c: med_std(q.assign(_="all"), c, "method", "_")["all"] for c in QUAL},
                  axis=1).loc[METHODS])

for c in ["accuracy", "bal_accuracy", "f1", "mcc"]:
    title(f"{c}: median ± std over seeds (rows: datasets by imbalance ratio)")
    display(med_std(q, c, "dataset", "method").loc[DATASETS, METHODS])

# %%
# Δ względem full_voting w pp: diverging, zero = biały
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
for ax, c in zip(axes, ["accuracy", "bal_accuracy"]):
    piv = (q.groupby(["dataset", "method"])[f"d_{c}"].median().unstack()
           .loc[DATASETS, [x for x in METHODS if x != REF]] * 100)
    lim = max(np.abs(piv.to_numpy()).max(), 0.1)
    sns.heatmap(piv, ax=ax, cmap="RdBu", center=0, vmin=-lim, vmax=lim, annot=True,
                fmt=".2f", annot_kws={"size": 7}, cbar_kws={"label": "pp"})
    ax.set_title(f"Δ {c} vs {REF} [pp], median over seeds")
    ax.set_xlabel("")
    ax.set_ylabel("")
show(fig)

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
others = [x for x in METHODS if x != REF]
for ax, c in zip(axes, ["accuracy", "bal_accuracy"]):
    sns.boxplot(data=q[q["method"] != REF].assign(v=lambda d: d[f"d_{c}"] * 100), x="v",
                y="method", order=others, ax=ax, color=BLUE, fliersize=2, linewidth=1)
    ax.axvline(0, color="0.2", lw=1)
    for e in (-EQ_MARGIN, EQ_MARGIN):
        ax.axvline(e * 100, color="0.5", lw=1, ls="--")
    ax.set_title(f"Δ {c} vs {REF} per run [pp] (dashed: ±{EQ_MARGIN * 100:g} pp)")
    ax.set_xlabel("pp")
    ax.set_ylabel("")
show(fig)

# %% [markdown]
# ## 6. Metryki kosztu (mediana ± std)
# Koszt obliczeniowy względem pełnego lasu: liczba ewaluowanych drzew na próbkę,
# udział drzew (trees_frac), oszczędzone ewaluacje, czas na próbkę i na użyte drzewo
# (narzut metody), koszt osobno dla klasy rzadkiej i częstej.

# %%
COST = ["trees_mean", "trees_median", "trees_max", "trees_frac", "saved_trees", "trees_min",
        "trees_maj", "ms_per_sample", "us_per_tree"]
for mode in MODES:
    title(f"cost per method, median ± std over runs ({mode})")
    dm = df[df["mode"] == mode].assign(_="all")
    display(pd.concat({c: med_std(dm, c, "method", "_", digits=3)["all"] for c in COST},
                      axis=1).loc[METHODS])

title(f"trees_frac: median ± std over seeds ({QUALITY_MODE})")
display(med_std(q, "trees_frac", "dataset", "method").loc[DATASETS, METHODS])

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
piv = q.groupby(["dataset", "method"])["trees_frac"].median().unstack().loc[DATASETS, EARLY]
sns.heatmap(piv, ax=axes[0], cmap="Blues", vmin=0, annot=True, fmt=".2f",
            annot_kws={"size": 7}, cbar_kws={"label": "share of T"})
axes[0].set_title(f"trees_frac (share of {T} trees), median over seeds")
axes[0].set_xlabel("")
axes[0].set_ylabel("")
# koszt na klasie rzadkiej / częstej: > 1 = rzadka klasa potrzebuje więcej drzew
ratio = (q.assign(r=q["trees_min"] / q["trees_maj"])
         .groupby(["dataset", "method"])["r"].median().unstack().loc[DATASETS, EARLY])
# kolor w skali log2 (2× więcej i 2× mniej drzew tak samo intensywne), w komórkach iloraz
lim = max(np.abs(np.log2(ratio.to_numpy())).max(), 0.5)
sns.heatmap(np.log2(ratio), ax=axes[1], cmap="RdBu_r", center=0, vmin=-lim, vmax=lim,
            annot=ratio, fmt=".2f", annot_kws={"size": 7},
            cbar_kws={"label": "log2(minority / majority)"})
axes[1].set_title("trees used: minority / majority class, median over seeds")
axes[1].set_xlabel("")
axes[1].set_ylabel("")
show(fig)

# rozkład n_used (ECDF, próbka z wszystkich przebiegów); pełny las i rf_sklearn = T zawsze
rng = np.random.default_rng(0)
nu = raw[(raw["mode"] == QUALITY_MODE) & raw["method"].isin(EARLY)]
fig, axes = plt.subplots(1, len(EARLY), figsize=(2.2 * len(EARLY), 3), sharey=True)
for ax, name in zip(np.atleast_1d(axes), EARLY):
    v = np.concatenate(nu.loc[nu["method"] == name, "n_used"].to_list())
    v = rng.choice(v, min(len(v), 50_000), replace=False)
    ax.ecdf(v, color=BLUE)
    ax.set_xscale("log")
    ax.set_xlim(1, T)
    ax.set_title(name, fontsize=8)
    ax.tick_params(labelsize=7)
np.atleast_1d(axes)[0].set_ylabel("ECDF")
fig.suptitle(f"trees used per sample (log), pooled over runs ({QUALITY_MODE})", fontsize=10)
show(fig)

# %% [markdown]
# ## 7. Zgodność z pełnym modelem (full_voting, hard voting)
# agree_full: odsetek próbek z tą samą klasą co pełny las; agree_full_bal: średnia
# zgodności w klasach prawdziwych (każda klasa waży tyle samo); agree_full_bal_ref: to samo
# w klasach przewidzianych przez pełny las; agree_full_min: zgodność na klasie rzadkiej.

# %%
AGREE = ["agree_full", "agree_full_bal", "agree_full_bal_ref", "agree_full_min"]
AGREE += ["agree_rf"] if HAS_RF else []
title(f"agreement per method, median ± std over runs ({QUALITY_MODE})")
display(pd.concat({c: med_std(q.assign(_="all"), c, "method", "_")["all"] for c in AGREE},
                  axis=1).loc[METHODS])
title("minimum over runs")
display(q.groupby("method")[AGREE].min().loc[METHODS])

for c in ["agree_full", "agree_full_bal", "agree_full_min"]:
    title(f"{c}: median ± std over seeds")
    display(med_std(q, c, "dataset", "method").loc[DATASETS, EARLY])

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
for ax, c in zip(axes, ["agree_full", "agree_full_min"]):
    piv = q.groupby(["dataset", "method"])[c].median().unstack().loc[DATASETS, EARLY]
    sns.heatmap(piv, ax=ax, cmap="Blues", vmin=min(0.9, np.nanmin(piv.to_numpy())), vmax=1,
                annot=True, fmt=".3f", annot_kws={"size": 7})
    ax.set_title(f"{c}, median over seeds")
    ax.set_xlabel("")
    ax.set_ylabel("")
show(fig)

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
for ax, c in zip(axes, ["agree_full", "agree_full_bal"]):
    sns.stripplot(data=q[q["method"].isin(EARLY)], x=c, y="method", order=EARLY, ax=ax,
                  color=BLUE, size=3, alpha=0.6, jitter=0.25)
    for p in q["pi"].dropna().unique():
        ax.axvline(p, color=ORANGE, lw=1.2, ls="--")
    ax.set_title(f"{c} per run (dashed: pi)")
    ax.set_ylabel("")
show(fig)

# %%
# zgodność parami wszystkich metod: średnia po przebiegach
pair_agree = pd.DataFrame(np.eye(len(METHODS)), index=METHODS, columns=METHODS)
for a, c in combinations(METHODS, 2):
    v = np.mean([(bq[(*r, a)] == bq[(*r, c)]).mean() for r in runs_q])
    pair_agree.loc[a, c] = pair_agree.loc[c, a] = v
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(pair_agree, ax=ax, cmap="Blues", vmin=pair_agree.to_numpy().min(), vmax=1,
            annot=True, fmt=".3f", annot_kws={"size": 6.5}, square=True)
ax.set_title(f"pairwise prediction agreement, mean over runs ({QUALITY_MODE})")
show(fig)

# %%
# anatomia niezgodności z pełnym lasem: kto miał rację na próbkach, gdzie metoda ≠ full
# (w binarnej jedna z nich zawsze ma rację). net_pp = wpływ na accuracy w pp
dis = q[q["method"] != REF].groupby("method")[["n", "n_dis", "b", "c", "dis_min"]].sum()
dis = dis.assign(
    dis_rate=dis["n_dis"] / dis["n"],
    method_right=dis["b"] / dis["n_dis"].where(dis["n_dis"] > 0),
    full_right=dis["c"] / dis["n_dis"].where(dis["n_dis"] > 0),
    on_minority=dis["dis_min"] / dis["n_dis"].where(dis["n_dis"] > 0),
    net_pp=(dis["b"] - dis["c"]) / dis["n"] * 100,
).loc[[x for x in METHODS if x != REF]]
title(f"disagreements with {REF}, pooled over runs ({QUALITY_MODE}): who was right")
display(dis)

# %% [markdown]
# ## 8. Szybkość (mediana ± std)
# time_s: mediana z n_repeats powtórzeń dla wszystkich próbek trybu; ms_per_sample
# i przyspieszenie są porównywalne między zbiorami o różnym n_test.

# %%
SPEED = ["ms_per_sample", "speedup_vs_full"] + (["speedup_vs_rf"] if HAS_RF else [])
for mode in MODES:
    dm = df[df["mode"] == mode].assign(_="all")
    title(f"speed per method, median ± std over runs ({mode})")
    display(pd.concat({c: med_std(dm, c, "method", "_", digits=3)["all"] for c in SPEED},
                      axis=1).loc[METHODS])
    for c in ["ms_per_sample", "speedup_vs_full"]:
        title(f"{c}: median ± std over seeds ({mode})")
        display(med_std(dm, c, "dataset", "method", digits=3).loc[DATASETS, METHODS])

# %%
fig, ax = plt.subplots(figsize=(12, 6))
sp = df[df["method"] != REF]
sns.boxplot(data=sp, x="speedup_vs_full", y="method", hue="mode", order=others,
            hue_order=MODES, ax=ax, palette=[BLUE, ORANGE][:len(MODES)], fliersize=2,
            linewidth=1, log_scale=True)
ax.axvline(1, color="0.2", lw=1)
ax.set_title(f"speedup vs {REF} per run (log; > 1 = faster)")
ax.set_ylabel("")
show(fig)

# %%
# kompromis szybkość / zgodność: mediany per metoda (punkty) i per zbiór (tło);
# front Pareto = metody, których żadna inna nie przewyższa w obu naraz
for mode in MODES:
    dm = df[(df["mode"] == mode) & df["method"].isin(EARLY)]
    per_ds = dm.groupby(["dataset", "method"])[["speedup_vs_full", "agree_full"]].median()
    per_m = dm.groupby("method")[["speedup_vs_full", "agree_full"]].median().loc[EARLY]
    sv, av = per_m["speedup_vs_full"].to_numpy(), per_m["agree_full"].to_numpy()
    pareto = [not any((sv[j] >= sv[i]) and (av[j] >= av[i]) and (sv[j] > sv[i] or
                      av[j] > av[i]) for j in range(len(sv))) for i in range(len(sv))]
    colors = [ORANGE if p else BLUE for p in pareto]
    fig, (ax0, ax) = plt.subplots(1, 2, figsize=(16, 6), width_ratios=[1, 1.3])
    # lewy: przegląd, mediana per zbiór (szare) i per metoda
    ax0.scatter(per_ds["speedup_vs_full"], per_ds["agree_full"], s=10, color="0.75")
    ax0.scatter(sv, av, s=40, color=colors, edgecolor="white", linewidth=1.5, zorder=3)
    ax0.set_title(f"per dataset (grey) and per method ({mode})", fontsize=10)
    # prawy: same metody, numery zamiast nazw (punkty leżą gęsto), nazwy w legendzie obok;
    # numery bliskich punktów jeden nad drugim
    ax.scatter(sv, av, s=40, color=colors, edgecolor="white", linewidth=1.5, zorder=3)
    placed: list[tuple[float, float]] = []
    y_span = av.max() - av.min() or 1e-3
    for i in np.argsort(av):
        near = sum(abs(np.log10(sv[i]) - px) < 0.06 and abs(av[i] - py) < 0.04 * y_span
                   for px, py in placed)
        placed.append((np.log10(sv[i]), av[i]))
        ax.annotate(str(i + 1), (sv[i], av[i]), xytext=(5 + 9 * near, 2),
                    textcoords="offset points", fontsize=8)
    ax.text(1.02, 1, "\n".join(f"{i + 1:>2}  {n}" for i, n in enumerate(EARLY)),
            transform=ax.transAxes, va="top", fontsize=8, family="monospace")
    ax.set_title("per method (orange = Pareto front)", fontsize=10)
    for a in (ax0, ax):
        a.set_xscale("log")
        a.set_xlabel(f"speedup vs {REF} (median)")
        a.set_ylabel("agree_full (median)")
    show(fig)
    print(f"[{mode}] Pareto front: {', '.join(n for n, p in zip(EARLY, pareto) if p)}")

# %% [markdown]
# ## 9. Testy statystyczne: czy metody są równoważne pełnemu lasowi
# Zawsze w trybie QUALITY_MODE (wszystkie próbki testowe). Uwaga: seedy to różne podziały
# tych samych danych, więc nie są w pełni niezależne; przy kilku seedach (plik testowy) testy
# po seedach mają małą moc: wnioski z pełnego pliku.

# %% [markdown]
# ### 9.1 Gwarancja zgodności: agree_full >= pi
# Dokładny test dwumianowy per przebieg, H0: P(zgodność) >= pi, H1: < pi; Holm per metoda.
# „raw_below” = agree_full < pi w próbce, „violations” = istotne po korekcie.

# %%
rows = []
for name, g in q[q["pi"].notna() & (q["method"] != REF)].groupby("method", sort=False):
    pi = g["pi"].iloc[0]
    p = [stats.binomtest(k, n, pi, alternative="less").pvalue
         for k, n in zip(g["n_agree"], g["n"])]
    pooled = stats.binomtest(int(g["n_agree"].sum()), int(g["n"].sum()), pi)
    ci = pooled.proportion_ci(confidence_level=1 - ALPHA, method="exact")
    rows.append({"method": name, "pi": pi, "runs": len(g),
                 "raw_below": int((g["agree_full"] < pi).sum()),
                 "violations": int((holm(np.array(p)) < ALPHA).sum()),
                 "min_p_holm": np.nanmin(holm(np.array(p))),
                 "pooled_agree": pooled.statistic, "ci_lo": ci.low, "ci_hi": ci.high})
pi_tests = pd.DataFrame(rows).set_index("method")
title("agree_full >= pi guarantee (one-sided binomial, Holm per method)")
display(pi_tests)

# %% [markdown]
# ### 9.2 McNemar per przebieg (metoda vs full_voting)
# Dokładny test na parach niezgodnych (b: rację ma metoda, c: pełny las), Holm per metoda.

# %%
q = q.assign(p_mcnemar=[stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
                        for b, c in zip(q["b"], q["c"])])
q["p_mcnemar_holm"] = q.groupby("method")["p_mcnemar"].transform(
    lambda p: holm(p.to_numpy()))
sig = q["p_mcnemar_holm"] < ALPHA
mcn = pd.DataFrame({
    "runs": q.groupby("method").size(),
    "significant": sig.groupby(q["method"]).sum(),
    "sig_better": (sig & (q["b"] > q["c"])).groupby(q["method"]).sum(),
    "sig_worse": (sig & (q["b"] < q["c"])).groupby(q["method"]).sum(),
    "median_p_holm": q.groupby("method")["p_mcnemar_holm"].median(),
}).loc[METHODS]
title("McNemar: method vs full_voting per run (Holm per method)")
display(mcn)

# %% [markdown]
# ### 9.3 TOST: równoważność accuracy i balanced accuracy z full_voting (±EQ_MARGIN)
# Per zbiór: różnice sparowane po seedach. Ogólnie: mediany różnic per zbiór (n = zbiory).

# %%
TOST_METRICS = ["accuracy", "bal_accuracy"]
per_ds = []
for (name, ds), g in q[q["method"] != REF].groupby(["method", "dataset"], sort=False):
    for c in TOST_METRICS:
        per_ds.append({"method": name, "dataset": ds, "metric": c,
                       **tost(g[f"d_{c}"].to_numpy(), EQ_MARGIN)})
per_ds = pd.DataFrame(per_ds)
for c in TOST_METRICS:
    title(f"TOST per dataset, {c} (±{EQ_MARGIN * 100:g} pp, pairs = seeds)")
    display(per_ds[per_ds["metric"] == c].pivot(index="dataset", columns="method",
                                                 values="verdict").loc[DATASETS, others])

overall = []
med_d = q[q["method"] != REF].groupby(["method", "dataset"])[
    [f"d_{c}" for c in TOST_METRICS]].median()
for name in others:
    for c in TOST_METRICS:
        r = tost(med_d.loc[name, f"d_{c}"].to_numpy(), EQ_MARGIN)
        n_eq = int(((per_ds["method"] == name) & (per_ds["metric"] == c)
                    & (per_ds["verdict"] == "equivalent")).sum())
        overall.append({"method": name, "metric": c, **r, "datasets_equiv": n_eq})
overall = pd.DataFrame(overall)
title(f"TOST over datasets (pairs = datasets, ±{EQ_MARGIN * 100:g} pp)")
display(overall.set_index(["metric", "method"]))

fig, axes = plt.subplots(1, len(TOST_METRICS), figsize=(14, 0.4 * len(others) + 1.5),
                         sharey=True)
for ax, c in zip(axes, TOST_METRICS):
    o = overall[overall["metric"] == c].set_index("method").loc[others]
    y = np.arange(len(o))[::-1]
    colors = [BLUE if v == "equivalent" else ORANGE for v in o["verdict"]]
    ax.axvspan(-EQ_MARGIN * 100, EQ_MARGIN * 100, color="0.92", zorder=0)
    ax.axvline(0, color="0.3", lw=1)
    ax.hlines(y, o["ci_lo"] * 100, o["ci_hi"] * 100, color=colors, lw=2)
    ax.scatter(o["mean"] * 100, y, color=colors, s=30, zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.set_yticks(y, o.index)
    ax.set_xlabel("pp")
    ax.set_title(f"Δ {c} vs {REF}: mean, 90% CI\n(grey band ±{EQ_MARGIN * 100:g} pp; "
                 f"blue = equivalent)", fontsize=10)
show(fig)

# %% [markdown]
# ### 9.4 Friedman + Nemenyi (Demšar 2006)
# Bloki = zbiory (mediana po seedach), metody bez duplikatów. Ranga 1 = najlepsza.

# %%
RANKED = [x for x in METHODS if x not in DUPLICATES]
RANK_TESTS = [("bal_accuracy", QUALITY_MODE, False), ("trees_frac", QUALITY_MODE, True)]
RANK_TESTS += [("ms_per_sample", mode, True) for mode in MODES]
mean_ranks = {}
for c, mode, lower_better in RANK_TESTS:
    piv = (df[(df["mode"] == mode) & df["method"].isin(RANKED)]
           .groupby(["dataset", "method"])[c].median().unstack()[RANKED])
    ranks = piv.rank(axis=1, ascending=lower_better).mean()
    stat, p = stats.friedmanchisquare(*piv.to_numpy().T)
    cd = critical_difference(len(RANKED), len(piv))
    mean_ranks[f"{c} ({mode})"] = ranks
    print(f"\nFriedman {c} ({mode}): chi2 = {stat:.2f}, p = {p:.2e}, N = {len(piv)} datasets, "
          f"k = {len(RANKED)}, CD = {cd:.2f}")
    cd_diagram(ranks, cd, f"{c} ({mode})")
mean_ranks = pd.DataFrame(mean_ranks)
title("mean ranks (1 = best)")
display(mean_ranks)

# %% [markdown]
# ### 9.5 Wilcoxon vs full_voting po zbiorach (Holm)
# Δ balanced accuracy: dwustronny; log(czas / czas full): jednostronny (metoda szybsza).

# %%
def wilcoxon_p(d: np.ndarray, alternative: str = "two-sided") -> float:
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    return stats.wilcoxon(d, zero_method="zsplit", alternative=alternative).pvalue


wil = pd.DataFrame(index=pd.Index(others, name="method"))
wil["d_bal_acc_med"] = med_d["d_bal_accuracy"].groupby("method").median()
wil["p_bal_acc"] = holm(np.array([wilcoxon_p(med_d.loc[x, "d_bal_accuracy"]) for x in others]))
for mode in MODES:
    t = df[df["mode"] == mode].pivot_table(index="dataset", columns="method",
                                           values="ms_per_sample", aggfunc="median")
    lr = np.log(t[others].div(t[REF], axis=0))
    wil[f"time_ratio_{mode}"] = np.exp(lr.median())
    wil[f"p_faster_{mode}"] = holm(np.array([wilcoxon_p(lr[x], "less") for x in others]))
title("Wilcoxon signed-rank vs full_voting over datasets (Holm per column)")
display(wil)

# %% [markdown]
# ## 10. Podsumowanie

# %%
summ = pd.DataFrame(index=pd.Index(others, name="method"))
for c in TOST_METRICS:
    o = overall[overall["metric"] == c].set_index("method")
    summ[f"d_{c}_pp"] = (o["mean"] * 100).round(3)
    summ[f"d_{c}_ci90"] = [f"[{lo * 100:.2f}, {hi * 100:.2f}]"
                           for lo, hi in zip(o["ci_lo"], o["ci_hi"])]
    summ[f"tost_{c}"] = o["verdict"]
    summ[f"equiv_ds_{c}"] = o["datasets_equiv"].astype(str) + f"/{len(DATASETS)}"
summ["mcnemar_sig_runs"] = (mcn["significant"].astype(str) + "/" + mcn["runs"].astype(str))
for c in ["agree_full", "agree_full_bal", "agree_full_min"]:
    summ[f"{c}_med"] = q.groupby("method")[c].median()
summ["agree_full_min_run"] = q.groupby("method")["agree_full"].min()
summ["pi_violations"] = pi_tests["violations"].astype("Int64")
for mode in MODES:
    dm = df[df["mode"] == mode].groupby("method")
    summ[f"speedup_{mode}"] = [f"{a:.2f} ± {b:.2f}" for a, b in zip(
        dm["speedup_vs_full"].median().loc[others], dm["speedup_vs_full"].std().loc[others])]
summ["trees_frac"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(
    q.groupby("method")["trees_frac"].median().loc[others],
    q.groupby("method")["trees_frac"].std().loc[others])]
summ = summ.join(mean_ranks.add_prefix("rank "))
title(f"summary vs {REF} ({RESULTS_FILE.name}, seeds {sorted(raw['seed'].unique().tolist())})")
display(summ.T)
