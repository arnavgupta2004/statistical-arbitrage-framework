from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.data.leakage import assert_no_lookahead
from statarb.signals.zscore import spread_zscore

from .test_hedge_ratio import pair


def brute(ly, lx, beta_t, window, t):
    """z at t with today's beta applied to the trailing window, from scratch."""
    sl = slice(t - window + 1, t + 1)
    s = ly.iloc[sl] - beta_t * lx.iloc[sl]
    return (s.iloc[-1] - s.mean()) / s.std(ddof=1)


def test_static_beta_matches_the_definition():
    ly, lx, _ = pair(300, seed=1)
    beta = pd.Series(0.8, index=ly.index)
    out = spread_zscore(ly, lx, beta, 60)
    for t in (59, 120, 299):
        assert out["z"].iloc[t] == pytest.approx(brute(ly, lx, 0.8, 60, t))
    assert out["z"].iloc[:59].isna().all()
    assert out["spread"].iloc[10] == pytest.approx(ly.iloc[10] - 0.8 * lx.iloc[10])


def test_time_varying_beta_applies_todays_hedge_to_the_whole_window():
    ly, lx, _ = pair(300, seed=2)
    beta = pd.Series(np.linspace(0.6, 1.1, 300), index=ly.index)
    out = spread_zscore(ly, lx, beta, 60)
    for t in (80, 200, 299):
        assert out["z"].iloc[t] == pytest.approx(brute(ly, lx, beta.iloc[t], 60, t))


def test_a_change_in_beta_does_not_create_a_spurious_jump_but_a_stitched_spread_does():
    ly, lx, _ = pair(400, seed=3)
    beta = pd.Series(0.8, index=ly.index)
    beta.iloc[250:] = 0.85  # a small hedge-ratio update
    z = spread_zscore(ly, lx, beta, 60)["z"]
    stitched = ly - beta * lx  # each day's own beta: jumps by 0.05 * lx ~ 0.2 on the update day
    naive = (stitched - stitched.rolling(60).mean()) / stitched.rolling(60).std()
    assert abs(z.iloc[250] - z.iloc[249]) < 1.5
    assert abs(naive.iloc[250] - naive.iloc[249]) > 4


def test_intercept_shift_and_scale_of_the_series_do_not_change_z():
    ly, lx, _ = pair(300, seed=4)
    beta = pd.Series(0.8, index=ly.index)
    base = spread_zscore(ly, lx, beta, 60)["z"]
    pd.testing.assert_series_equal(
        base, spread_zscore(ly + 5.0, lx, beta, 60)["z"]
    )  # alpha cancels
    pd.testing.assert_series_equal(base, spread_zscore(ly, lx + 3.0, beta, 60)["z"])


def test_expanding_window_and_min_periods():
    ly, lx, _ = pair(200, seed=5)
    beta = pd.Series(0.8, index=ly.index)
    out = spread_zscore(ly, lx, beta, None, min_periods=30)
    assert out["z"].iloc[:29].isna().all() and out["z"].iloc[29:].notna().all()
    s = ly.iloc[:150] - 0.8 * lx.iloc[:150]
    assert out["z"].iloc[149] == pytest.approx((s.iloc[-1] - s.mean()) / s.std(ddof=1))


def test_degenerate_windows_are_nan_not_inf():
    idx = pd.bdate_range("2020-01-01", periods=80)
    lx = pd.Series(np.linspace(4, 5, 80), index=idx)
    ly = 2 + 0.5 * lx  # spread with beta=0.5 is exactly constant: zero variance
    out = spread_zscore(ly, lx, pd.Series(0.5, index=idx), 30)
    assert out["z"].isna().all() and not np.isinf(out["z"]).any()


@pytest.mark.parametrize("window", [30, None])
def test_zscore_is_causal_even_with_a_time_varying_beta(window):
    ly, lx, _ = pair(400, seed=6, drift_sd=0.003)
    frame = pd.DataFrame({"ly": ly, "lx": lx, "beta": 0.8 + 0.2 * np.sin(np.arange(400) / 40)})
    fn = lambda f: spread_zscore(f["ly"], f["lx"], f["beta"], window, 20)[["z"]]  # noqa: E731
    # beta is part of the frame here, so the perturbation test also proves z ignores future beta
    assert_no_lookahead(fn, frame, frame.index[[100, 200, 300, 380]])
