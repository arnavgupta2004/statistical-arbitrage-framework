"""ADF: exact agreement with statsmodels, plus properties that hold analytically."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller

from statarb.selection.adf import adf_test, ols, trend_matrix


def random_walk(n, seed, drift=0.0):
    return np.cumsum(np.random.default_rng(seed).normal(drift, 1.0, n))


def ar1(n, phi, seed, burn=200):
    e = np.random.default_rng(seed).normal(size=n + burn)
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def sm(x, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return adfuller(x, result_object=False, **kw)


@pytest.mark.parametrize("regression", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("autolag", ["aic", "bic", None])
@pytest.mark.parametrize("series", ["rw", "ar", "rw_drift"])
def test_matches_statsmodels_exactly(regression, autolag, series):
    x = {"rw": random_walk(350, 1), "ar": ar1(350, 0.85, 2), "rw_drift": random_walk(350, 3, 0.1)}[
        series
    ]
    ours = adf_test(x, regression=regression, autolag=autolag)
    ref = sm(x, regression=regression, autolag=autolag)
    assert ours.statistic == pytest.approx(ref[0], abs=1e-9)
    assert ours.pvalue == pytest.approx(ref[1], abs=1e-8)  # p-value surface amplifies 1e-9 in stat
    assert ours.usedlag == ref[2] and ours.nobs == ref[3]
    for k, v in ref[4].items():
        assert ours.critical_values[k] == pytest.approx(v)


@pytest.mark.parametrize("maxlag", [0, 1, 4, 8])
def test_fixed_maxlag_matches_statsmodels(maxlag):
    x = ar1(400, 0.9, 11)
    ours = adf_test(x, regression="c", maxlag=maxlag, autolag=None)
    ref = sm(x, regression="c", maxlag=maxlag, autolag=None)
    assert ours.statistic == pytest.approx(ref[0], abs=1e-9) and ours.usedlag == maxlag == ref[2]


def test_stationary_series_rejects_and_random_walk_does_not():
    assert adf_test(ar1(500, 0.5, 4)).rejects_unit_root()
    assert not adf_test(random_walk(500, 5)).rejects_unit_root()


def test_statistic_is_invariant_to_shift_and_scale_with_a_constant():
    """Adding a constant or rescaling cannot change a regression t-ratio (analytic property)."""
    x = ar1(300, 0.8, 6)
    base = adf_test(x, regression="c", autolag=None, maxlag=3).statistic
    assert adf_test(
        5.0 * x + 100.0, regression="c", autolag=None, maxlag=3
    ).statistic == pytest.approx(base)
    # with regression="n" a shift is NOT harmless: the model has no intercept to absorb it
    assert adf_test(x + 50.0, regression="n", autolag=None, maxlag=3).statistic != pytest.approx(
        adf_test(x, regression="n", autolag=None, maxlag=3).statistic
    )


def test_size_under_a_true_unit_root_is_near_nominal():
    """Monte Carlo: 5 % nominal size.  600 draws -> s.e. ~0.9 pp; allow 3 s.e. + finite-sample slack."""
    rng = np.random.default_rng(2024)
    rejections = sum(
        adf_test(np.cumsum(rng.normal(size=250))).rejects_unit_root() for _ in range(600)
    )
    assert 0.02 <= rejections / 600 <= 0.085


def test_power_against_a_persistent_stationary_alternative():
    rng = np.random.default_rng(7)
    rejections = 0
    for _ in range(200):
        e = rng.normal(size=450)
        x = np.zeros(450)
        for t in range(1, 450):
            x[t] = 0.9 * x[t - 1] + e[t]
        rejections += adf_test(x[200:]).rejects_unit_root()
    assert rejections / 200 > 0.6  # phi=0.9, T=250: high but not certain -- power is finite


def test_a_level_shift_is_read_as_a_unit_root():
    """Documented weakness: a break to a new mean makes a stationary series look non-stationary."""
    rng = np.random.default_rng(8)
    rejections = 0
    for _ in range(200):
        x = np.zeros(300)
        e = rng.normal(size=300)
        for t in range(1, 300):
            x[t] = 0.5 * x[t - 1] + e[t]
        x[150:] += 6.0
        rejections += adf_test(x).rejects_unit_root()
    assert rejections / 200 < 0.3  # would be ~100 % without the break


@pytest.mark.parametrize(
    "bad, msg",
    [
        (np.ones(50), "constant"),
        (np.array([1.0, np.nan, 2.0] * 30), "NaN"),
        (np.arange(4.0), "short"),
    ],
)
def test_degenerate_inputs_are_rejected(bad, msg):
    with pytest.raises(ValueError, match=msg):
        adf_test(bad)


def test_bad_arguments():
    x = ar1(100, 0.5, 1)
    with pytest.raises(ValueError):
        adf_test(x, regression="bogus")
    with pytest.raises(ValueError):
        adf_test(x, autolag="hqic")
    with pytest.raises(ValueError):
        adf_test(x, maxlag=999)
    with pytest.raises(ValueError):
        adf_test(np.ones((5, 2)))


def test_ols_helper_recovers_coefficients_and_flags_collinearity():
    rng = np.random.default_rng(0)
    x = np.column_stack([np.ones(100), rng.normal(size=100)])
    y = x @ [2.0, -3.0] + rng.normal(scale=0.1, size=100)
    beta, resid, ssr, xtx_inv = ols(y, x)
    assert beta == pytest.approx([2.0, -3.0], abs=0.05) and ssr == pytest.approx(resid @ resid)
    assert xtx_inv == pytest.approx(np.linalg.inv(x.T @ x))
    with pytest.raises(ValueError, match="collinear"):
        ols(y, np.column_stack([x, x[:, 1]]))
    assert trend_matrix(4, "ct").tolist() == [[1, 1], [1, 2], [1, 3], [1, 4]]
    assert trend_matrix(3, "n").shape == (3, 0)
