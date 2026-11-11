"""Predykcje out-of-bag: każdą próbkę uczącą oceniają tylko drzewa, które jej nie
widziały przy uczeniu. Daje to nieobciążone głosy lasu bez osobnego zbioru
walidacyjnego (korzystają z nich Hyper i MarginEarlyStop)."""
import numpy as np


def probka_uczaca(rf, X_train, n_max=None, seed=0):
    """Indeksy (rosnąco) co najwyżej n_max losowych wierszy X_train; None = wszystkie.
    Sprawdza przy okazji, że X_train to dane, na których uczono rf (z bootstrapem)."""
    n = len(X_train)
    if not rf.bootstrap:
        raise ValueError("głosy out-of-bag wymagają lasu z bootstrap=True")
    if getattr(rf, "_n_samples", n) != n:
        raise ValueError(f"X_train ma {n} wierszy, a las uczono na {rf._n_samples}")
    if n_max is None or n_max >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, n_max, replace=False))


def predykcje_oob(rf, X_train, wiersze):
    """Generator po drzewach lasu (w kolejności rf.estimators_). Dla każdego drzewa:

        oob   pozycje w wiersze, których drzewo NIE widziało przy uczeniu,
        pred  indeksy klas przewidziane przez drzewo dla tych próbek.
    """
    X_train = np.asarray(X_train)
    n = len(X_train)
    X = X_train[wiersze]
    # indeksy bootstrapu każdego drzewa; generator sklearn nie trzyma ich naraz w pamięci
    if hasattr(rf, "_get_estimators_indices"):
        indeksy = rf._get_estimators_indices()
    else:
        indeksy = rf.estimators_samples_
    for tree, w_probie in zip(rf.estimators_, indeksy):
        oob = np.flatnonzero(np.bincount(w_probie, minlength=n)[wiersze] == 0)
        yield oob, tree.predict(X[oob]).astype(int)


def glosy_oob(rf, X_train, n_prior=None, seed=0):
    """Głosy out-of-bag: votes[n, i] = ile drzew, które NIE widziały próbki n
    przy uczeniu, głosuje na klasę i. X_train to dane, na których uczono rf.

    n_prior ogranicza liczbę (losowo wybranych) próbek uczących; None = wszystkie.
    """
    wiersze = probka_uczaca(rf, X_train, n_prior, seed)
    votes = np.zeros((len(wiersze), len(rf.classes_)), dtype=int)
    for oob, pred in predykcje_oob(rf, X_train, wiersze):
        votes[oob, pred] += 1
    return votes
