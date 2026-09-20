"""Rolling statistics (causality, equality with brute force) and the Hurst estimators (known cases)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.data.leakage import assert_no_lookahead, truncation_invariance
from statarb.models.rolling import ewm_mean_std, rolling_beta, rolling_mean_std, zscore
from statarb.selection.hurst import hurst_rs, hurst_variance


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    return pd.Series(np.cumsum(rng.normal(size=300)) + 100, index=idx)


@pytest.fixture
def frame(series):
    rng = np.random.default_rng(1)
    return pd.DataFrame({"y": series, "x": series * 0.5 + rng.normal(size=300)}, index=series.index)


def test_rolling_mean_std_equals_brute_force(series):
    mean, std = rolling_mean_std(series, 20)
    for t in (19, 50, 299):
        w = series.iloc[t - 19 : t + 1]
        assert mean.iloc[t] == pytest.approx(w.mean()) and std.iloc[t] == pytest.approx(
            w.std(ddof=1)
        )
    assert mean.iloc[:19].isna().all() and mean.iloc[19:].notna().all()  # no partial windows


def test_expanding_window_and_min_periods(series):
    mean, std = rolling_mean_std(series, None, min_periods=5)
    assert mean.iloc[:4].isna().all() and mean.iloc[4] == pytest.approx(series.iloc[:5].mean())
    assert std.iloc[-1] == pytest.approx(series.std(ddof=1))
    m10, _ = rolling_mean_std(series, 20, min_periods=10)
    assert m10.iloc[:9].isna().all() and m10.iloc[9] == pytest.approx(series.iloc[:10].mean())


def test_zscore_by_hand_and_exclusion_of_the_current_observation(series):
    z = zscore(series, 20)
    w = series.iloc[30:50]
    assert z.iloc[49] == pytest.approx((series.iloc[49] - w.mean()) / w.std(ddof=1))
    zx = zscore(series, 20, exclude_current=True)
    prior = series.iloc[29:49]  # the 20 rows *before* row 49
    assert zx.iloc[49] == pytest.approx((series.iloc[49] - prior.mean()) / prior.std(ddof=1))
    assert abs(zx.iloc[49]) != pytest.approx(abs(z.iloc[49]))


def test_zscore_of_a_flat_window_is_nan_not_infinite():
    """A zero-variance window has no scale.  With the current row *inside* the window the numerator is
    also 0 (0/0 = NaN regardless), so the case that needs the guard is a flat *prior* window followed
    by a move (``exclude_current=True``): without it the result would be +/-inf."""
    flat_then_jump = pd.Series([5.0] * 30 + [5.5] + list(np.linspace(5, 6, 10)))
    z = zscore(flat_then_jump, 10, exclude_current=True)
    assert np.isnan(z.iloc[30])  # x - mean = 0.5, std = 0 -> undefined, not inf
    assert not np.isinf(z).any()
    assert np.isnan(zscore(flat_then_jump, 10).iloc[15])


def test_rolling_beta_equals_window_ols(frame):
    beta, alpha = rolling_beta(frame["y"], frame["x"], 60)
    for t in (59, 150, 299):
        w = frame.iloc[t - 59 : t + 1]
        b, a = np.polyfit(w["x"], w["y"], 1)
        assert beta.iloc[t] == pytest.approx(b) and alpha.iloc[t] == pytest.approx(a)
    eb, _ = rolling_beta(frame["y"], frame["x"], None, min_periods=30)
    assert eb.iloc[-1] == pytest.approx(np.polyfit(frame["x"], frame["y"], 1)[0])


def test_ewm(series):
    m, s = ewm_mean_std(series, halflife=10)
    assert m.iloc[0] != m.iloc[0] and m.iloc[-1] == pytest.approx(
        series.ewm(halflife=10).mean().iloc[-1]
    )
    assert s.iloc[-1] > 0


CUTS = [60, 120, 200, 280]


@pytest.mark.parametrize("exclude", [False, True])
def test_rolling_functions_pass_the_lookahead_detectors(series, exclude):
    cuts = series.index[CUTS]
    assert_no_lookahead(lambda s: zscore(s, 30, exclude_current=exclude), series, cuts)
    assert_no_lookahead(lambda s: zscore(s, None, min_periods=10), series, cuts)
    assert_no_lookahead(lambda s: rolling_mean_std(s, 25)[0], series, cuts)
    assert_no_lookahead(lambda s: ewm_mean_std(s, 8)[1], series, cuts)


def test_rolling_beta_passes_the_lookahead_detectors(frame):
    cuts = frame.index[CUTS]
    assert_no_lookahead(lambda f: rolling_beta(f["y"], f["x"], 40)[0].to_frame("b"), frame, cuts)
    assert_no_lookahead(
        lambda f: rolling_beta(f["y"], f["x"], None, 20)[1].to_frame("a"), frame, cuts
    )


def test_the_detectors_would_catch_a_centred_window(series):
    """Negative control: the same harness flags a window that peeks forward."""
    leaky = lambda s: s.rolling(21, center=True).mean()  # noqa: E731
    assert not truncation_invariance(leaky, series, series.index[CUTS]).passed


# ---- Hurst ---------------------------------------------------------------------------------
def fbm(n, h, seed):
    """Fractional Brownian motion by Cholesky of the exact fGn covariance."""
    k = np.arange(n)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * h) - 2 * np.abs(k) ** (2 * h) + np.abs(k - 1) ** (2 * h))
    cov = gamma[np.abs(k[:, None] - k[None, :])]
    return np.cumsum(
        np.linalg.cholesky(cov + 1e-10 * np.eye(n)) @ np.random.default_rng(seed).normal(size=n)
    )


@pytest.mark.parametrize("h", [0.3, 0.5, 0.7])
def test_variance_estimator_recovers_h_for_fractional_brownian_motion(h):
    est = np.mean([hurst_variance(fbm(1000, h, s), max_lag=30) for s in range(6)])
    assert est == pytest.approx(h, abs=0.06)


def test_variance_estimator_on_a_random_walk_is_half():
    est = np.mean(
        [
            hurst_variance(np.cumsum(np.random.default_rng(s).normal(size=3000)), 50)
            for s in range(10)
        ]
    )
    assert est == pytest.approx(0.5, abs=0.04)


def test_variance_estimator_reads_a_fast_mean_reverting_series_as_anti_persistent():
    from statarb.models.ou import simulate_ou

    ests = [
        hurst_variance(simulate_ou(0.5, 0.0, 1.0, 3000, rng=np.random.default_rng(s)), 50)
        for s in range(6)
    ]
    assert np.mean(ests) < 0.2  # saturating variance over the lag window => far below 0.5


def test_rescaled_range_is_biased_upward_for_iid_noise_as_documented():
    est = np.mean([hurst_rs(np.random.default_rng(s).normal(size=1000)) for s in range(20)])
    assert (
        0.5 < est < 0.65
    )  # not 0.5: the small-sample bias that makes H > 0.5 uninformative on its own


def test_hurst_input_validation():
    with pytest.raises(ValueError):
        hurst_variance(np.arange(50.0), max_lag=50)
    with pytest.raises(ValueError):
        hurst_rs(np.arange(10.0))
