"""
Porównanie metod inferencji dla lasu losowego:
pełny RandomForestClassifier ze sklearn, pełne głosowanie wszystkich drzew
oraz metody przyspieszonej inferencji (np. IB pruning).

Nową metodę dodaje się jako podklasę InferenceMethod z metodami
predict_batch i predict_one, a następnie przez methods_rf.add_method(...).
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from generate_table_IB import czy_stop

MODES = ("batch", "online")


# ---------------------------------------------------------------------------
# metody inferencji
# ---------------------------------------------------------------------------
class InferenceMethod:
    """Wspólny interfejs metod inferencji.

    predict_batch(X) -> (y_pred, n_used)   dla macierzy X (n_próbek, n_cech)
    predict_one(x)   -> (y, n_used)        dla jednej próbki x o kształcie (1, n_cech)
    n_used = liczba drzew odpytanych dla danej próbki.
    """

    name = "method"

    def fit(self, rf):
        self.rf = rf
        self.trees = rf.estimators_
        self.classes = rf.classes_
        self.T = len(self.trees)
        self.l = len(self.classes)
        return self

    def predict_batch(self, X):
        raise NotImplementedError

    def predict_one(self, x):
        raise NotImplementedError


class SklearnRF(InferenceMethod):
    """Oryginalny rf.predict (uśrednia prawdopodobieństwa drzew)."""

    name = "rf_sklearn"

    def predict_batch(self, X):
        return self.rf.predict(X), np.full(len(X), self.T)

    def predict_one(self, x):
        return self.rf.predict(x)[0], self.T


class FullVoting(InferenceMethod):
    """Pełne głosowanie większościowe: każde drzewo oddaje jeden głos."""

    name = "full_voting"

    def predict_batch(self, X):
        votes = np.zeros((len(X), self.l), dtype=int)
        wiersze = np.arange(len(X))
        for tree in self.trees:
            votes[wiersze, tree.predict(X).astype(int)] += 1
        return self.classes[votes.argmax(axis=1)], np.full(len(X), self.T)

    def predict_one(self, x):
        votes = np.zeros(self.l, dtype=int)
        for tree in self.trees:
            votes[int(tree.predict(x)[0])] += 1
        return self.classes[votes.argmax()], self.T


def czy_stop_wektorowo(votes, prog):
    """czy_stop dla wielu próbek naraz; votes ma kształt (n_próbek, l)."""
    t = -np.sort(-votes, axis=1)                    # głosy malejąco w każdym wierszu
    return t[:, 0] >= prog[tuple(t[:, 1:].T)]


class IBPruning(InferenceMethod):
    """Statystyczny instance-based pruning (Hernández-Lobato i in., 2009):
    drzewa są pytane po kolei, a głosowanie kończy się, gdy wynik jest pewny
    z prawdopodobieństwem >= pi (według tablicy progów prog)."""

    def __init__(self, prog, name="ib_pruning", pi=None):
        self.prog = np.asarray(prog)
        self.name = name
        self.pi = pi

    @classmethod
    def from_npz(cls, path, name=None):
        d = np.load(path)
        pi = float(d["pi"])
        return cls(d["prog"], name=name or f"ib_pi{round(pi * 100, 6):g}", pi=pi)

    def fit(self, rf):
        super().fit(rf)
        T_prog, l_prog = self.prog.shape[0] - 1, self.prog.ndim + 1
        if (T_prog, l_prog) != (self.T, self.l):
            raise ValueError(f"{self.name}: tablica jest dla T={T_prog}, l={l_prog}, "
                             f"a las ma T={self.T}, l={self.l}")
        return self

    def predict_batch(self, X):
        votes = np.zeros((len(X), self.l), dtype=int)
        n_used = np.full(len(X), self.T)
        aktywne = np.arange(len(X))
        for k, tree in enumerate(self.trees):
            votes[aktywne, tree.predict(X[aktywne]).astype(int)] += 1
            stop = czy_stop_wektorowo(votes[aktywne], self.prog)
            n_used[aktywne[stop]] = k + 1
            aktywne = aktywne[~stop]
            if len(aktywne) == 0:
                break
        return self.classes[votes.argmax(axis=1)], n_used

    def predict_one(self, x):
        votes = np.zeros(self.l, dtype=int)
        for k, tree in enumerate(self.trees):
            votes[int(tree.predict(x)[0])] += 1
            if czy_stop(votes, self.prog):
                break
        return self.classes[votes.argmax()], k + 1


# ---------------------------------------------------------------------------
# zbiór metod dla jednego lasu
# ---------------------------------------------------------------------------
class methods_rf:
    """Zbiór metod inferencji dla jednego wytrenowanego lasu losowego.

    Po run(X, mode) wyniki są w macierzach:
        Y_pred[mode]  (n_metod, n_próbek)  predykcje
        N_used[mode]  (n_metod, n_próbek)  liczba odpytanych drzew
        times[mode]   (n_metod,)           czas predykcji w s (mediana z n_repeats)
    Wiersz i odpowiada metodzie names[i].
    """

    def __init__(self, rf, methods=None):
        self.rf = rf
        self.classes = rf.classes_
        self.T = len(rf.estimators_)
        self.l = len(rf.classes_)
        self.methods = []
        for m in (methods if methods is not None else [SklearnRF(), FullVoting()]):
            self.add_method(m)
        self.Y_pred, self.N_used, self.times = {}, {}, {}

    @property
    def names(self):
        return [m.name for m in self.methods]

    def add_method(self, method):
        if method.name in self.names:
            raise ValueError(f"metoda o nazwie '{method.name}' już istnieje")
        self.methods.append(method.fit(self.rf))
        return self

    def index(self, name):
        return self.names.index(name)

    def _run_method(self, method, X, mode):
        if mode == "batch":
            return method.predict_batch(X)
        wyniki = [method.predict_one(x[None, :]) for x in X]
        y, n = zip(*wyniki)
        return np.array(y), np.array(n)

    def run(self, X, mode="batch", n_repeats=1):
        if mode not in MODES:
            raise ValueError(f"mode musi być jednym z {MODES}")
        Y = np.empty((len(self.methods), len(X)), dtype=self.classes.dtype)
        N = np.empty((len(self.methods), len(X)), dtype=int)
        times = np.empty(len(self.methods))
        for i, method in enumerate(self.methods):
            pomiary = []
            for _ in range(n_repeats):
                start = time.perf_counter()
                y, n = self._run_method(method, X, mode)
                pomiary.append(time.perf_counter() - start)
            Y[i], N[i], times[i] = y, n, np.median(pomiary)
        self.Y_pred[mode], self.N_used[mode], self.times[mode] = Y, N, times
        return Y, N, times


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
class benchmark:
    """Uruchamia wszystkie metody z methods_rf w dwóch scenariuszach:
    'batch'  - cały X_test naraz,
    'online' - pierwsze n_online próbek X_test, każda osobno,
    i porównuje je metrykami jakości, zgodnością z rf_sklearn / full_voting
    oraz czasem predykcji."""

    REF_RF = SklearnRF.name
    REF_FULL = FullVoting.name

    def __init__(self, models, X_test, y_test, metadata=None, n_online=500, n_repeats=1):
        self.models = models
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test)
        self.n_online = min(n_online, len(self.X_test))
        self.n_repeats = n_repeats
        self.metadata = dict(metadata or {})
        self.metadata.update(
            T=models.T, l=models.l, n_test=len(self.X_test), n_online=self.n_online,
            n_repeats=n_repeats, methods=models.names,
            date=datetime.now().isoformat(timespec="seconds"),
        )
        self.Y_pred, self.N_used, self.times = {}, {}, {}

    def _X_y(self, mode):
        if mode == "batch":
            return self.X_test, self.y_test
        return self.X_test[: self.n_online], self.y_test[: self.n_online]

    def run(self, modes=MODES):
        for mode in modes:
            X, _ = self._X_y(mode)
            Y, N, t = self.models.run(X, mode=mode, n_repeats=self.n_repeats)
            self.Y_pred[mode], self.N_used[mode], self.times[mode] = Y, N, t
        return self

    def _ref(self, name):
        """Indeks metody odniesienia (albo None, jeśli jej nie ma)."""
        names = self.models.names
        return names.index(name) if name in names else None

    def results(self, mode="batch"):
        if mode not in self.Y_pred:
            raise RuntimeError(f"brak wyników dla '{mode}', najpierw wywołaj run()")
        Y, N, times = self.Y_pred[mode], self.N_used[mode], self.times[mode]
        _, y_true = self._X_y(mode)
        i_rf, i_full = self._ref(self.REF_RF), self._ref(self.REF_FULL)
        n = Y.shape[1]

        rows = []
        for i, name in enumerate(self.models.names):
            y = Y[i]
            rows.append({
                "method": name,
                "accuracy": accuracy_score(y_true, y),
                "precision": precision_score(y_true, y, average="macro", zero_division=0),
                "recall": recall_score(y_true, y, average="macro", zero_division=0),
                "f1": f1_score(y_true, y, average="macro", zero_division=0),
                "agree_rf": (y == Y[i_rf]).mean() if i_rf is not None else np.nan,
                "agree_full": (y == Y[i_full]).mean() if i_full is not None else np.nan,
                "time_s": times[i],
                "ms_per_sample": times[i] / n * 1e3,
                "speedup_vs_rf": times[i_rf] / times[i] if i_rf is not None else np.nan,
                "speedup_vs_full": times[i_full] / times[i] if i_full is not None else np.nan,
                "trees_mean": N[i].mean(),
                "trees_median": np.median(N[i]),
                "trees_max": N[i].max(),
                "trees_frac": N[i].mean() / self.models.T,
            })
        return pd.DataFrame(rows).set_index("method")

    def summary(self):
        print("metadane:")
        for k, v in self.metadata.items():
            print(f"  {k}: {v}")
        with pd.option_context("display.width", 200, "display.max_columns", None,
                               "display.float_format", "{:.4f}".format):
            for mode in self.Y_pred:
                n = self.Y_pred[mode].shape[1]
                print(f"\n[{mode}] ({n} próbek)")
                print(self.results(mode))

    def to_dict(self):
        d = {"metadata": json.dumps(self.metadata, default=str),
             "names": np.array(self.models.names),
             "classes": self.models.classes,
             "y_test": self.y_test}
        for mode in self.Y_pred:
            d[f"Y_pred_{mode}"] = self.Y_pred[mode]
            d[f"N_used_{mode}"] = self.N_used[mode]
            d[f"times_{mode}"] = self.times[mode]
        return d

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_dict())
        return path


# ---------------------------------------------------------------------------
# wiele modeli / zbiorów / powtórzeń
# ---------------------------------------------------------------------------
class benchmark_suite:
    """Zbiór wyników benchmark dla wielu modeli (np. wiele zbiorów danych
    i kilka powtórzeń z różnymi seedami).

    Każdy benchmark jest identyfikowany wartościami kluczy z jego metadanych
    (domyślnie ('dataset', 'seed')). Ostatni klucz to powtórzenie, po którym
    uśrednia aggregate()."""

    SUMMARY_COLS = ["accuracy", "f1", "agree_rf", "agree_full", "ms_per_sample",
                    "speedup_vs_rf", "speedup_vs_full", "trees_mean", "trees_frac"]

    def __init__(self, keys=("dataset", "seed")):
        self.keys = list(keys)
        self.runs = []

    def add(self, b):
        brak = [k for k in self.keys if k not in b.metadata]
        if brak:
            raise ValueError(f"w metadanych benchmarku brakuje kluczy: {brak}")
        self.runs.append(b)
        return self

    def __len__(self):
        return len(self.runs)

    def results(self, mode="batch"):
        """Wyniki wszystkich benchmarków: jeden wiersz na (klucze..., metoda)."""
        frames = []
        for b in self.runs:
            r = b.results(mode).reset_index()
            for k in reversed(self.keys):
                r.insert(0, k, b.metadata[k])
            frames.append(r)
        return pd.concat(frames, ignore_index=True).set_index([*self.keys, "method"])

    def aggregate(self, mode="batch", stats=("mean", "std"), overall=False):
        """Statystyki po powtórzeniach (ostatni klucz) dla każdego zbioru i metody.
        overall=True uśrednia dodatkowo po wszystkich zbiorach (wiersz na metodę)."""
        df = self.results(mode).reset_index()
        group = ["method"] if overall else [*self.keys[:-1], "method"]
        df = df.drop(columns=[k for k in self.keys if k not in group])
        stats = stats if isinstance(stats, str) else list(stats)
        return df.groupby(group, sort=False).agg(stats)

    def summary(self, cols=None):
        cols = cols or self.SUMMARY_COLS
        modes = [m for m in MODES if all(m in b.Y_pred for b in self.runs)]
        with pd.option_context("display.width", 250, "display.max_columns", None,
                               "display.max_rows", None,
                               "display.float_format", "{:.4f}".format):
            for mode in modes:
                print(f"\n[{mode}] średnia z powtórzeń ({self.keys[-1]}) dla każdego zbioru:")
                print(self.aggregate(mode, stats="mean")[cols])
                print(f"\n[{mode}] średnia po wszystkich zbiorach i powtórzeniach:")
                print(self.aggregate(mode, stats="mean", overall=True)[cols])

    def save(self, out_dir):
        """Zapisuje macierze każdego benchmarku (.npz) oraz tabele wyników (.csv)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for b in self.runs:
            nazwa = "_".join(f"{k}{b.metadata[k]}" if k != "dataset" else str(b.metadata[k])
                             for k in self.keys)
            b.save(out_dir / "runs" / f"{nazwa}.npz")
        for mode in MODES:
            if all(mode in b.Y_pred for b in self.runs):
                self.results(mode).to_csv(out_dir / f"results_{mode}.csv")
                self.aggregate(mode).to_csv(out_dir / f"aggregate_{mode}.csv")
        return out_dir
