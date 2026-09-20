"""Engle-Granger: agreement with statsmodels.coint and the reason it needs its own critical values."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from statarb.selection.adf import adf_test
from statarb.selection.cointegration import engle_granger, engle_granger_both, spread

from .test_adf import ar1


def pair(n, seed, beta=2.0, rho=0.8, noise=1.0, alpha=1.0):
    """Cointegrated by construction: Y = alpha + beta X + AR(rho) noise, X a random walk."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n)) + 50
    return alpha + beta * x + noise * ar1(n, rho, seed + 1000), x


def independent_walks(n, seed):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n)), np.cumsum(rng.normal(size=n))


@pytest.mark.parametrize("trend", ["c", "ct", "n"])
@pytest.mark.parametrize("autolag", ["aic", "bic", None])
def test_matches_statsmodels_coint(trend, autolag):
    y, x = pair(300, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = coint(y, x, trend=trend, autolag=autolag)
    ours = engle_granger(y, x, trend=trend, autolag=autolag)
    assert ours.statistic == pytest.approx(ref[0], abs=1e-9)
    assert ours.pvalue == pytest.approx(ref[1], abs=1e-12)
    if trend != "n":
        assert [ours.critical_values[k] for k in ("1%", "5%", "10%")] == pytest.approx(list(ref[2]))


def test_matches_statsmodels_with_several_regressors():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(size=(400, 2)), axis=0)
    y = 1.5 * x[:, 0] - 0.7 * x[:, 1] + ar1(400, 0.7, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = coint(y, x)
    ours = engle_granger(y, x)
    assert ours.statistic == pytest.approx(ref[0], abs=1e-9) and ours.pvalue == pytest.approx(
        ref[1]
    )
    assert ours.hedge_ratio == pytest.approx([1.5, -0.7], abs=0.15)


def test_recovers_the_hedge_ratio_and_intercept_and_the_spread():
    y, x = pair(2000, 5, beta=2.0, alpha=1.0)
    r = engle_granger(y, x)
    assert r.hedge_ratio[0] == pytest.approx(2.0, abs=0.01) and r.intercept == pytest.approx(
        1.0, abs=0.6
    )
    assert r.spread == pytest.approx(y - r.intercept - r.hedge_ratio[0] * x)
    assert spread(y, x, r.hedge_ratio, r.intercept) == pytest.approx(r.spread)
    assert r.rejects_no_cointegration() and r.rsquared > 0.99


def test_rejects_for_cointegrated_pairs_and_not_for_independent_walks():
    assert all(engle_granger(*pair(300, s)).rejects_no_cointegration() for s in range(10))
    fails = sum(
        engle_granger(*independent_walks(300, s)).rejects_no_cointegration() for s in range(50)
    )
    assert fails <= 8  # 5 % nominal; a handful of false positives is expected, most are not


def test_size_is_near_nominal_with_the_cointegration_critical_values():
    rng = np.random.default_rng(99)
    rej = sum(
        engle_granger(
            np.cumsum(rng.normal(size=250)), np.cumsum(rng.normal(size=250))
        ).rejects_no_cointegration()
        for _ in range(500)
    )
    assert 0.02 <= rej / 500 <= 0.085


def test_ordinary_adf_critical_values_on_the_residual_over_reject_badly():
    """The reason engle_granger uses MacKinnon's *cointegration* tables, demonstrated.

    OLS picks beta to make the residual look stationary, so its ADF statistic is systematically
    lower than under the plain Dickey-Fuller null.  Treating it as an ordinary ADF test rejects far
    more often than 5 % on independent random walks (the spurious-regression trap).
    """
    rng = np.random.default_rng(2025)
    naive = correct = 0
    for _ in range(400):
        y, x = np.cumsum(rng.normal(size=250)), np.cumsum(rng.normal(size=250))
        res = engle_granger(y, x)
        naive += adf_test(np.asarray(res.spread), regression="n").pvalue < 0.05
        correct += res.pvalue < 0.05
    assert naive / 400 > 0.15 > 0.085 > correct / 400


def test_both_directions_differ_in_finite_samples_and_count_as_two_tests():
    y, x = pair(120, 8, rho=0.9, noise=3.0)
    fwd, rev = engle_granger_both(y, x)
    assert fwd.n_tests == rev.n_tests == 2 and engle_granger(y, x).n_tests == 1
    assert fwd.statistic == pytest.approx(engle_granger(y, x).statistic)
    assert rev.statistic == pytest.approx(engle_granger(x, y).statistic)
    assert fwd.statistic != pytest.approx(rev.statistic, abs=1e-6)  # the asymmetry is real
    assert fwd.hedge_ratio[0] * rev.hedge_ratio[0] <= 1.0 + 1e-12  # slopes multiply to R^2 <= 1


def test_pandas_index_is_preserved():
    y, x = pair(300, 2)
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = engle_granger(pd.Series(y, index=idx), pd.Series(x, index=idx))
    assert isinstance(r.spread, pd.Series) and r.spread.index.equals(idx)
    assert isinstance(spread(pd.Series(y, index=idx), x, r.hedge_ratio, r.intercept), pd.Series)


def test_invalid_inputs():
    y, x = pair(100, 1)
    with pytest.raises(ValueError, match="same number"):
        engle_granger(y, x[:-1])
    with pytest.raises(ValueError, match="NaN"):
        engle_granger(np.r_[np.nan, y[1:]], x)
    with pytest.raises(ValueError, match="collinear"):
        engle_granger(3 * x + 1, x)
    with pytest.raises(ValueError):
        engle_granger(y, x, trend="bogus")
