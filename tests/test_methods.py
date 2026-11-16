"""Metody klasyfikacyjne (src/methods): FastTree i pętle, IB pruning, HYPER, GLEE, margines,
metoda własna i głosy out-of-bag. Każda reguła stopu jest porównana z wzorem liczonym
niezależnie od kodu albo z zależnością opisaną w implementacja.md / changes.md."""
import math
from itertools import product

import numpy as np
import pytest
from conftest import (T, all_vote_states, check_batch_equals_online, full_vote,
                      tree_index_predictions)
from scipy.stats import betabinom, norm
from sklearn.ensemble import RandomForestClassifier

from methods import (FullVoting, Glee, Hyper, IBPruning, MarginEarlyStop, Own, generate_table,
                     ib_pruning)
from methods.base import fast_trees
from methods.glee import alpha_from_pi, default_min_votes, glee_table
from methods.hyper import (MAX_CELLS, hyper_table, pair_prior, pair_table, smooth_valid,
                           stop_hyper, stop_hyper_pair, uniform_prior)
from methods.ib_pruning import load_table, p_same, stop_ib, stop_ib_one
from methods.margin import (calibration_predictions, margin_drop, n_trees_from_margin,
                            vote_margin, votes_from_predictions)
from methods.oob import oob_votes, sample_rows


# ---------------------------------------------------------------------------------------
# base.py, voting.py, oob.py
# ---------------------------------------------------------------------------------------

def test_fast_tree_matches_sklearn(three):
    X = three.X_test
    for tree, exp in zip(fast_trees(three.rf), tree_index_predictions(three.rf, X)):
        np.testing.assert_array_equal(tree.predict(np.asarray(X, dtype=np.float32)), exp)


@pytest.mark.parametrize("bad", [np.zeros(8), np.zeros((3, 7))])
def test_x32_rejects_wrong_shape(binary, bad):
    # FastTree nie waliduje wejścia: zły kształt bez tej kontroli daje śmieciowe wyniki
    with pytest.raises(ValueError, match="expected"):
        FullVoting().fit(binary.rf).predict_batch(bad)


def test_full_voting_equals_direct_vote(three):
    y, n = check_batch_equals_online(FullVoting().fit(three.rf), three.X_test)
    np.testing.assert_array_equal(y, full_vote(three.rf, three.X_test))
    assert (n == T).all()
    assert y.dtype == three.rf.classes_.dtype       # etykiety, a nie indeksy klas


def test_shuffled_order_does_not_modify_shared_tree_list(binary):
    shared = list(fast_trees(binary.rf))
    m = IBPruning(pi=0.95, order="shuffled", seed=3).fit(binary.rf)
    assert fast_trees(binary.rf) == shared          # lista wspólna dla innych metod bez zmian
    assert m.trees != shared and sorted(map(id, m.trees)) == sorted(map(id, shared))


def test_oob_votes_equal_direct_computation(three):
    """oob_votes korzysta z rf._get_estimators_indices (prywatne); porównanie z publicznym
    estimators_samples_ wyłapie zmianę w nowej wersji scikit-learn."""
    n = len(three.X_train)
    pred = tree_index_predictions(three.rf, three.X_train)
    expected = np.zeros((n, 3), dtype=int)
    for p, in_bag in zip(pred, three.rf.estimators_samples_):
        oob = np.bincount(in_bag, minlength=n) == 0
        expected[np.flatnonzero(oob), p[oob]] += 1
    np.testing.assert_array_equal(oob_votes(three.rf, three.X_train, None), expected)
    rows = sample_rows(three.rf, three.X_train, 100, seed=0)
    np.testing.assert_array_equal(oob_votes(three.rf, three.X_train, 100), expected[rows])


# ---------------------------------------------------------------------------------------
# IB pruning (ib_pruning.py)
# ---------------------------------------------------------------------------------------

@pytest.fixture
def table_dir(tmp_path, monkeypatch):
    """generate_table i dobór tablicy w fit() piszą/czytają w katalogu tymczasowym."""
    monkeypatch.setattr(ib_pruning, "TABLE_DIR", tmp_path)
    return tmp_path


@pytest.mark.parametrize("t0, t1, T_", [(3, 1, 11), (0, 0, 9), (2, 7, 25),
                                        (2, 2, 10), (3, 1, 12)])     # parzyste T: remis możliwy
def test_ib_p_same_matches_beta_binomial(t0, t1, T_):
    """Dla dwóch klas urna Polyi z jednostajnym a priori daje rozkład beta-dwumianowy
    pozostałych głosów: niezależne sprawdzenie rekurencji (także remisu jako przegranej)."""
    rest = T_ - t0 - t1
    k = np.arange(rest + 1)                         # głosy lidera wśród pozostałych
    lead, other = max(t0, t1), min(t0, t1)
    pmf = betabinom.pmf(k, rest, lead + 1, other + 1)
    expected = pmf[(lead + k) > (other + rest - k)].sum()
    assert p_same((t0, t1), T_) == pytest.approx(expected, abs=1e-12)


def test_ib_stored_table_matches_regenerated(table_dir):
    """Plik z repozytorium jest tym, co liczy generate_table (zmiana rekurencji albo
    konwencji remisu rozjechałaby oba)."""
    stored, pi = load_table(ib_pruning.Path(__file__).resolve().parents[1]
                            / "data" / "tablice" / "T51_L2_pi95.npz")
    fresh, pi_fresh = load_table(generate_table(51, 2, 0.95))
    np.testing.assert_array_equal(fresh, stored)
    assert pi == pi_fresh == 0.95


def test_ib_generated_table_is_minimal_threshold(table_dir):
    """threshold[rest] = najmniejsze t1 >= t2, przy którym p_same >= pi (T + 1: brak),
    a reguła jest monotoniczna w t1 (inaczej sam próg nie wystarczałby do decyzji)."""
    T_, pi = 13, 0.9
    threshold, _ = load_table(generate_table(T_, 3, pi))
    for t2 in range(T_ + 1):
        for t3 in range(min(t2, T_ - t2) + 1):
            ok = [t1 for t1 in range(t2, T_ - t2 - t3 + 1) if p_same((t1, t2, t3), T_) >= pi]
            assert threshold[t2, t3] == (ok[0] if ok else T_ + 1)
            if ok:
                assert ok == list(range(ok[0], T_ - t2 - t3 + 1))


def test_ib_table_for_other_forest_rejected(binary, three):
    with pytest.raises(ValueError, match="table is for T=501"):
        IBPruning(table="T501_L2_pi98.npz").fit(binary.rf)
    with pytest.raises(ValueError, match="l=2"):        # tablica binarna, las 3-klasowy
        IBPruning(table="T51_L2_pi95.npz").fit(three.rf)


@pytest.mark.parametrize("order", ["fixed", "shuffled"])
def test_ib_batch_equals_online(binary, order):
    m = IBPruning(pi=0.95, order=order, seed=1).fit(binary.rf)
    _, n = check_batch_equals_online(m, binary.X_test)
    assert ((n >= 1) & (n <= T)).all()
    assert n.mean() < T / 2                         # wczesny stop naprawdę działa


@pytest.mark.parametrize("order", ["fixed", "random"])
def test_ib_samples_that_used_all_trees_equal_full_voting(binary, order):
    y, n = IBPruning(pi=0.95, order=order, seed=0).fit(binary.rf).predict_batch(binary.X_test)
    full = full_vote(binary.rf, binary.X_test)
    np.testing.assert_array_equal(y[n == T], full[n == T])
    assert (y == full).mean() >= 0.95 - 0.03        # agree_full blisko pi


def test_ib_stopped_sample_has_settled_vote(binary):
    """Pętla batch: po n_used drzewach stan głosów spełnia regułę, a po n_used - 1 jeszcze nie."""
    m = IBPruning(pi=0.95).fit(binary.rf)
    y, n = m.predict_batch(binary.X_test)
    pred = np.stack([t.predict(m._X32(binary.X_test)) for t in m.trees])
    for i in range(len(y)):
        votes = np.bincount(pred[: n[i], i], minlength=2)
        assert binary.rf.classes_[votes.argmax()] == y[i]
        if n[i] < T:
            assert stop_ib_one(votes, m.threshold)
        prev = np.bincount(pred[: n[i] - 1, i], minlength=2)
        assert n[i] == 1 or not stop_ib_one(prev, m.threshold)


def test_ib_pair_equals_full_for_two_classes(binary):
    full = IBPruning(pi=0.95).fit(binary.rf).predict_batch(binary.X_test)
    pair = IBPruning(pi=0.95, multiclass="pair").fit(binary.rf).predict_batch(binary.X_test)
    np.testing.assert_array_equal(full[0], pair[0])
    np.testing.assert_array_equal(full[1], pair[1])


# ---------------------------------------------------------------------------------------
# HYPER (hyper.py)
# ---------------------------------------------------------------------------------------

def hyper_brute_force_stop(t: np.ndarray, prior: np.ndarray, pi: float) -> bool:
    """Stop wprost z modelu urny (wzór 3 artykułu): posterior składu ∝ ∏ C(T_i, t_i) · P(T);
    stop, gdy P(lider ściśle wygra pełne głosowanie) >= pi albo reszta głosów nic nie zmieni."""
    l, T_ = len(t), prior.shape[0] - 1
    c = int(np.argmax(t))                           # remis -> niższy indeks, jak _leader
    if t[c] - np.delete(t, c).max() > T_ - int(t.sum()):
        return True
    g = h = 0.0
    for head in product(range(T_ + 1), repeat=l - 1):
        comp = (*head, T_ - sum(head))
        if comp[-1] < 0 or any(ci < ti for ci, ti in zip(comp, t)):
            continue
        w = prior[head] * np.prod([math.comb(ci, int(ti)) for ci, ti in zip(comp, t)])
        h += w
        if comp[c] > max(np.delete(comp, c)):
            g += w
    return h > 0 and g >= pi * h


def test_hyper_smoothing_keeps_prior_nonnegative_and_mass_mode_keeps_mass():
    rng = np.random.default_rng(1)
    hist = np.zeros((52, 52))
    valid = np.indices(hist.shape).sum(axis=0) <= 51
    cells = np.argwhere(valid)
    for i, j in cells[rng.choice(len(cells), 200)]:
        hist[i, j] += rng.integers(1, 50)
    hist[51, 0] = hist[0, 51] = 1000                # jednomyślne składy przy brzegu
    mass = smooth_valid(hist, valid, 5, "mass")
    assert mass.sum() == pytest.approx(hist.sum())
    for out in (mass, smooth_valid(hist, valid, 5, "mean")):
        assert (out >= 0).all()                     # ujemne wartości dawały nan w tablicy
        assert (out[~valid] == 0).all()


def test_hyper_uniform_prior_table_equals_ib_table():
    """Jednostajne a priori = IB pruning: ta sama decyzja w każdym stanie głosowania."""
    ib, pi = load_table("T51_L2_pi95.npz")
    votes = all_vote_states(T, 2)
    assert len(votes) == 1325                       # liczba stanów z changes.md
    np.testing.assert_array_equal(stop_hyper(votes, hyper_table(uniform_prior(T, 2), pi)),
                                  stop_ib(votes, ib))


@pytest.mark.parametrize("sparse", [False, True])
def test_hyper_table_matches_brute_force_three_classes(sparse):
    T_, pi = 12, 0.9
    rng = np.random.default_rng(5)
    prior = rng.random((T_ + 1, T_ + 1))
    if sparse:
        prior[rng.random(prior.shape) < 0.7] = 0
    prior[np.indices(prior.shape).sum(axis=0) > T_] = 0
    prior /= prior.sum()
    votes = all_vote_states(T_, 3)
    assert len(votes) == 363
    expected = np.array([hyper_brute_force_stop(v, prior, pi) for v in votes])
    np.testing.assert_array_equal(stop_hyper(votes, hyper_table(prior, pi)), expected)


def test_hyper_pair_table_requires_strict_lead():
    prior = pair_prior(np.random.default_rng(3).integers(0, 15, size=(300, 3)), T)
    threshold = pair_table(prior, 0.9)
    assert (threshold >= np.arange(T + 1) + 1).all()   # remis w głosach nigdy nie zatrzymuje
    assert not stop_hyper_pair(np.array([[5, 5, 0], [0, 7, 7]]), threshold).any()


def test_hyper_fit_validates_calibration_data(binary):
    with pytest.raises(ValueError, match="training data"):
        Hyper(pi=0.95).fit(binary.rf)
    with pytest.raises(ValueError, match="validation set"):
        Hyper(pi=0.95, calib="val").fit(binary.rf, binary.X_train, binary.y_train)
    # pomyłkowo podany inny zbiór: indeksy bootstrapu nie pasowałyby do wierszy
    with pytest.raises(ValueError, match="trained on"):
        Hyper(pi=0.95).fit(binary.rf, binary.X_test)
    rf = RandomForestClassifier(n_estimators=5, bootstrap=False, random_state=0)
    rf.fit(binary.X_train, binary.y_train)
    with pytest.raises(ValueError, match="bootstrap=True"):
        Hyper(pi=0.95).fit(rf, binary.X_train)
    Hyper(pi=0.95, calib="val").fit(rf, X_val=binary.X_val)    # z val bootstrap niepotrzebny


def test_hyper_too_many_states_rejected_but_pair_works(five):
    assert (T + 1) ** 4 > MAX_CELLS
    with pytest.raises(ValueError, match="too many states"):
        Hyper(pi=0.95).fit(five.rf, five.X_train)
    m = Hyper(pi=0.95, multiclass="pair").fit(five.rf, five.X_train)
    _, n = check_batch_equals_online(m, five.X_test)
    assert n.mean() < T


@pytest.mark.parametrize("multiclass, calib", [("full", "oob"), ("pair", "val")])
def test_hyper_three_classes_table_is_valid_and_stops_early(three, multiclass, calib):
    """Regresja błędu z changes.md: ujemne ~1e-20 po wygładzaniu dawały nan w tablicy
    i HYPER przy 3 klasach przestawał przerywać (352 zamiast 47 drzew z 501)."""
    m = Hyper(pi=0.95, multiclass=multiclass, calib=calib).fit(
        three.rf, three.X_train, three.y_train, three.X_val, three.y_val)
    assert np.isfinite(m.prior).all() and (m.prior >= 0).all()
    y, n = check_batch_equals_online(m, three.X_test)
    assert n.mean() < 0.6 * T
    full = full_vote(three.rf, three.X_test)
    assert (y == full).mean() >= 0.9
    np.testing.assert_array_equal(y[n == T], full[n == T])


def test_hyper_pair_equals_full_for_two_classes_and_odd_T(binary):
    """changes.md: dla dwóch klas i nieparzystego T reguła pary daje to samo co pełna tablica."""
    full = Hyper(pi=0.95).fit(binary.rf, binary.X_train).predict_batch(binary.X_test)
    pair = Hyper(pi=0.95, multiclass="pair").fit(binary.rf, binary.X_train).predict_batch(
        binary.X_test)
    np.testing.assert_array_equal(full[0], pair[0])
    np.testing.assert_array_equal(full[1], pair[1])


# ---------------------------------------------------------------------------------------
# GLEE (glee.py)
# ---------------------------------------------------------------------------------------

def glee_stop(t1: int, t2: int, m: int, alpha: float, min_votes: int, fpc: bool,
              one_sided: bool) -> bool:
    """Reguła z rozdz. II-C artykułu COMET, liczona skalarnie dla jednego stanu."""
    n = t1 + t2
    if n < min_votes or n == 0 or n > m:
        return False
    p = t1 / n
    z = norm.ppf(1 - (alpha if one_sided else alpha / 2))
    delta = z * math.sqrt(p * (1 - p) / n)
    if fpc and n > 0.05 * m:
        delta *= math.sqrt((m - n) / (m - 1))
    return p - delta > 0.5


def test_glee_alpha_and_default_min_votes():
    # 1 - 0.99 = 0.010000000000000009 bez zaokrąglenia
    assert [alpha_from_pi(p) for p in (0.99, 0.98, 0.999)] == [0.01, 0.02, 0.001]
    assert [default_min_votes(alpha_from_pi(p)) for p in (0.95, 0.99, 0.999, 0.9999)] == [
        15, 15, 30, 45]


@pytest.mark.parametrize("m", [51, 501])
@pytest.mark.parametrize("alpha", [0.05, 0.001])
@pytest.mark.parametrize("fpc, one_sided", [(True, True), (False, False)])
def test_glee_table_equals_formula(m, alpha, fpc, one_sided):
    min_votes = default_min_votes(alpha)
    threshold = glee_table(m, alpha, None, fpc, one_sided)
    assert len(threshold) == m // 2 + 1
    for t2 in range(0, m // 2 + 1, max(1, m // 60)):
        for t1 in range(t2, m - t2 + 1):
            expected = glee_stop(t1, t2, m, alpha, min_votes, fpc, one_sided)
            assert (t1 >= threshold[t2]) == expected, (t1, t2)


def test_glee_no_stop_before_min_votes(five):
    _, n = check_batch_equals_online(Glee(pi=0.95, order="fixed").fit(five.rf), five.X_test)
    assert n.min() >= 15
    assert n.mean() < T


def test_glee_random_order_reproducible_and_agrees_with_full(binary):
    a = Glee(pi=0.95).fit(binary.rf).predict_batch(binary.X_test)
    b = Glee(pi=0.95).fit(binary.rf).predict_batch(binary.X_test)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    full = full_vote(binary.rf, binary.X_test)
    np.testing.assert_array_equal(a[0][a[1] == T], full[a[1] == T])
    assert (a[0] == full).mean() >= 0.93


# ---------------------------------------------------------------------------------------
# Margines (margin.py)
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("l", [2, 5])
def test_margin_drop_equals_brute_force(l):
    rng = np.random.default_rng(l)
    pred = rng.integers(0, l, size=(15, 40))
    pred[rng.random(pred.shape) < 0.4] = -1         # drzewo widziało próbkę: brak głosu
    pred[:3] = rng.integers(0, l, size=(3, 40))     # co najmniej trzy głosy na próbkę
    y = rng.integers(0, l, 40)
    result = margin_drop(pred, y, l)
    base = vote_margin(votes_from_predictions(pred, l), y)
    for i in range(len(pred)):
        voting = pred[i] >= 0
        mg = vote_margin(votes_from_predictions(np.delete(pred, i, axis=0), l), y)
        expected = (base - mg)[voting].mean() if voting.any() else -np.inf
        assert result[i] == pytest.approx(expected)


@pytest.mark.parametrize("mu, sigma, k, expected", [
    (0.5, 0.2, 0.0, 26),        # 51 * 0.5 = 25.5 -> w górę
    (0.5, 0.2, -10, 1),         # ujemne -> co najmniej jedno drzewo
    (0.9, 0.2, 5, T),           # ponad 1 -> wszystkie
])
def test_margin_n_trees(mu, sigma, k, expected):
    assert n_trees_from_margin(T, mu, sigma, k) == expected


def test_margin_calibration_maps_labels_to_indices(three):
    pred, y = calibration_predictions(three.rf, three.X_train, three.y_train, None)
    # te same głosy co oob_votes (wiersze z co najmniej dwoma drzewami out-of-bag)
    votes = oob_votes(three.rf, three.X_train)
    ok = votes.sum(axis=1) >= 2
    np.testing.assert_array_equal(votes_from_predictions(pred, 3), votes[ok])
    # indeksy klas, a nie etykiety 3, 7, 10
    np.testing.assert_array_equal(y, np.searchsorted(three.rf.classes_, three.y_train)[ok])


@pytest.mark.parametrize("ranking", ["mean_d_m", "accuracy", "none"])
def test_margin_predicts_with_first_n_star_ranked_trees(five, ranking):
    m = MarginEarlyStop(k=-0.5, ranking=ranking).fit(five.rf, five.X_train, five.y_train)
    y, n = check_batch_equals_online(m, five.X_test)
    assert (n == m.n_star).all() and 1 <= m.n_star <= T
    np.testing.assert_array_equal(y, full_vote(five.rf, five.X_test, m.order[: m.n_star]))
    assert sorted(m.order) == list(range(T))        # ranking to permutacja drzew


def test_margin_n_star_monotone_in_k_and_large_k_is_full_voting(three):
    m = MarginEarlyStop(k=0).fit(three.rf, three.X_train, three.y_train)
    name = m.name
    n_stars = [m.set_k(k).n_star for k in (-3, -1, -0.5, 0, 0.5, 1, 3)]
    assert n_stars == sorted(n_stars)
    m.set_k(100)
    assert m.n_star == T and m.name == name         # set_k nie zmienia nazwy
    y, _ = m.predict_batch(three.X_test)
    np.testing.assert_array_equal(y, full_vote(three.rf, three.X_test))


# ---------------------------------------------------------------------------------------
# Metoda własna (own.py, klasyfikacja)
# ---------------------------------------------------------------------------------------

def fit_all(method, case):
    return method.fit(case.rf, case.X_train, case.y_train, case.X_val, case.y_val)


@pytest.mark.parametrize("multiclass", ["full", "pair"])
def test_own_lam_one_is_hyper(three, multiclass):
    own = fit_all(Own(pi=0.95, lam=1.0, multiclass=multiclass), three)
    hyper = fit_all(Hyper(pi=0.95, multiclass=multiclass), three)
    np.testing.assert_array_equal(own.threshold, hyper.threshold)


def test_own_lam_zero_is_ib_pruning(binary):
    a = fit_all(Own(pi=0.95, lam=0.0), binary).predict_batch(binary.X_test)
    b = IBPruning(pi=0.95).fit(binary.rf).predict_batch(binary.X_test)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("case, multiclass", [("three", "full"), ("five", "pair")])
def test_own_glee_variant_stops_at_first_of_both_rules(request, case, multiclass):
    """changes.md: z glee = true liczba drzew to minimum z HYPER-a (own bez glee) i GLEE."""
    c = request.getfixturevalue(case)
    plain = fit_all(Own(pi=0.95, lam=0.5, multiclass=multiclass), c)
    merged = fit_all(Own(pi=0.95, lam=0.5, glee=True, pi_glee=0.9, min_votes=8,
                         multiclass=multiclass), c)
    glee = Glee(pi=0.9, min_votes=8, order="fixed").fit(c.rf)
    _, n_plain = plain.predict_batch(c.X_test)
    _, n_glee = glee.predict_batch(c.X_test)
    _, n = check_batch_equals_online(merged, c.X_test)
    np.testing.assert_array_equal(n, np.minimum(n_plain, n_glee))
