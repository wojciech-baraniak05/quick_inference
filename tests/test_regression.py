"""Metody regresyjne (src/methods/regression.py, own.py): warunek stopu = wzory (2), (6),
(10)-(11) dokumentu na wyjściach drzew, konwencje progów, t_min, delta, wariancje out-of-bag."""
import numpy as np
import pytest
from conftest import T, check_batch_equals_online
from scipy.stats import t as student
from sklearn.ensemble import RandomForestRegressor

from methods import GleeRegression, IBRegression, OwnRegression
from methods.base import fast_reg_trees
from methods.oob import oob_variances
from methods.regression import student_multiplier, sum_sq_threshold


def direct_stop(outputs: np.ndarray, c, delta: float, nu0: float = 0.0,
                sigma0_2: float = 0.0) -> tuple[float, int]:
    """Pierwsze t, dla którego c[t] * sigma_t <= delta; bez stopu: średnia wszystkich drzew."""
    T_ = len(outputs)
    for t in range(1, T_):
        y = outputs[:t]
        dof = nu0 + t - 1
        if not np.isfinite(c[t]) or dof <= 0:
            continue
        sigma = np.sqrt((nu0 * sigma0_2 + ((y - y.mean()) ** 2).sum()) / dof)
        if c[t] * sigma <= delta:
            return y.mean(), t
    return outputs.mean(), T_


def test_student_multiplier():
    c = student_multiplier(T, 0.95, t_min=3)
    assert np.isinf(c[:3]).all() and c[T] == 0
    t = 10
    assert c[t] == pytest.approx(student.ppf(0.975, t - 1) * np.sqrt((T - t) / (T * t)))
    assert np.isinf(student_multiplier(T, 0.95, t_min=1)[1])      # brak stopni swobody
    assert np.isfinite(student_multiplier(T, 0.95, t_min=1, nu0=2)[1])


def test_sum_sq_threshold_conventions():
    c = student_multiplier(T, 0.95, t_min=4)
    thr = sum_sq_threshold(c, delta=2.0)
    assert (thr[:4] == -1).all()                    # S_t >= 0 nie spełni warunku
    assert thr[T] == np.inf                         # po wszystkich drzewach zawsze koniec
    assert thr[10] == pytest.approx((2.0 / c[10]) ** 2 * 9)
    thr_prior = sum_sq_threshold(c, 2.0, nu0=5, sigma0_2=100.0)
    assert thr_prior[10] == pytest.approx((2.0 / c[10]) ** 2 * 14 - 500)


@pytest.mark.parametrize("make", [
    lambda: IBRegression(pi=0.9, delta_std=0.1, t_min=6),
    lambda: GleeRegression(pi=0.9, delta_std=0.1, t_min=3),
    lambda: GleeRegression(pi=0.9, delta_std=0.1, t_min=3, fpc=False),   # c[T] > 0
    lambda: OwnRegression(pi=0.9, delta_std=0.1, nu0=5),
])
def test_stop_rule_equals_formula(regression, make):
    m = make().fit(regression.rf, regression.X_train, regression.y_train)
    y, n = check_batch_equals_online(m, regression.X_test)
    X = m._X32(regression.X_test)
    out = np.stack([t.predict(X) for t in m.trees], axis=1)
    for i in range(len(out)):
        y_exp, n_exp = direct_stop(out[i], m.multiplier, m.delta, m.nu0, m.sigma0_2)
        assert n[i] == n_exp
        assert y[i] == pytest.approx(y_exp, rel=1e-9, abs=1e-9)
    assert 1 < n.mean() < T                         # przypadek nietrywialny


def test_identical_trees_do_not_stop_before_t_min(regression):
    """implementacja.md 11.2: S_t = 0 (identyczne wyjścia) nie może zatrzymać próbki
    przed t_min."""
    rf = RandomForestRegressor(n_estimators=20, random_state=0).fit(regression.X_train,
                                                                     regression.y_train)
    rf.estimators_ = [rf.estimators_[0]] * 20       # 20 identycznych drzew: S_t = 0
    X = regression.X_test[:40]
    for m, t_min in ((IBRegression(delta=0.1, t_min=5), 5),
                     (GleeRegression(pi=0.99, delta=0.1), 15)):
        y, n = m.fit(rf).predict_batch(X)
        assert (n == t_min).all(), m.name
        np.testing.assert_allclose(y, rf.predict(X))


def test_delta_std_is_converted_in_fit(regression):
    m = IBRegression(delta_std=0.2).fit(regression.rf, y_train=regression.y_train)
    assert m.delta == pytest.approx(0.2 * np.std(regression.y_train))
    with pytest.raises(ValueError, match="zero standard deviation"):
        IBRegression(delta_std=0.2).fit(regression.rf, y_train=np.ones(10))


def test_oob_variances_equal_direct(regression):
    rf = regression.rf
    X = np.asarray(regression.X_train, dtype=np.float32)
    outs = np.stack([t.predict(X) for t in fast_reg_trees(rf)])
    oob = np.stack([np.bincount(s, minlength=len(X)) == 0 for s in rf.estimators_samples_])
    expected = [np.var(outs[oob[:, i], i], ddof=1) for i in range(len(X))
                if oob[:, i].sum() >= 2]
    np.testing.assert_allclose(oob_variances(rf, regression.X_train), expected,
                               rtol=1e-9, atol=1e-9)
