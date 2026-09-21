"""Causal regimes and the regime Sharpe-difference test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.data.leakage import assert_no_lookahead
from statarb.statistics.regimes import (
    causal_regimes,
    expanding_above_median,
    regime_sharpe_difference,
)


def test_expanding_median_by_hand_and_never_uses_the_future():
    x = pd.Series([1.0, 3.0, 2.0, 10.0, 0.0, 5.0])
    out = expanding_above_median(x, min_periods=3)
    # medians through t: nan, nan, 2, 2.5, 2, 2.5 -> x > median
    assert out.isna().tolist() == [True, True, False, False, False, False]
    assert out.dropna().tolist() == [0.0, 1.0, 0.0, 1.0]
    y = x.copy()
    y.iloc[5] = -100.0
    assert expanding_above_median(y, 3).iloc[:5].equals(out.iloc[:5]) or True
    assert np.array_equal(
        expanding_above_median(y, 3).iloc[:5].to_numpy(), out.iloc[:5].to_numpy(), equal_nan=True
    )


def _market(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    vol = np.where((np.arange(n) // 150) % 2 == 0, 0.006, 0.02)
    return pd.DataFrame(
        {"mkt": rng.normal(0.0003, vol), "disp": np.abs(rng.normal(0.012, 0.003, n))}, index=idx
    )


def test_regime_labels_use_only_information_through_the_previous_day():
    df = _market()

    def fn(x: pd.DataFrame) -> pd.DataFrame:
        return causal_regimes(x["mkt"], x["disp"], window=21, trend_window=60, min_periods=100)

    cuts = [df.index[i] for i in (150, 300, 450, 700)]
    assert_no_lookahead(fn, df, cuts)
    # today's own return cannot move today's label
    p = df.copy()
    p.iloc[400, 0] += 0.3
    a, b = fn(df), fn(p)
    assert a.iloc[:401].equals(b.iloc[:401]) and not a.equals(b)


def test_volatility_regime_flips_after_the_volatility_does():
    df = _market()
    r = causal_regimes(df["mkt"], df["disp"], window=21, trend_window=60, min_periods=100)
    stress, calm_again = r["vol_high"].iloc[190:299], r["vol_high"].iloc[335:449]
    # the threshold is the expanding median, so regimes are relative: high once turbulence has
    # been running for a window, low again once calm returns (blocks: calm, stress, calm)
    assert stress.mean() > 0.9 and calm_again.mean() < 0.3
    assert r["vol_high"].iloc[:100].isna().all() and set(r["trend_up"].dropna().unique()) <= {
        0.0,
        1.0,
    }


def test_sharpe_difference_size_and_power():
    rng = np.random.default_rng(5)
    n = 500
    rej = 0
    for i in range(200):
        r = rng.normal(0.0004, 0.01, n)
        lab = (rng.random(n) < 0.5).astype(float)
        rej += (
            regime_sharpe_difference(r, lab, n_boot=300, seed=i, mean_block=1.0)["p_value"] < 0.05
        )
    assert 0.01 <= rej / 200 <= 0.11  # no difference: ~5 %
    hits = 0
    for i in range(60):
        lab = (rng.random(n) < 0.5).astype(float)
        r = rng.normal(0.0, 0.01, n) + 0.006 * lab  # a Sharpe of ~9.5 annualised in one state only
        hits += (
            regime_sharpe_difference(r, lab, n_boot=300, seed=i, mean_block=1.0)["p_value"] < 0.05
        )
    assert hits / 60 > 0.9


def test_sharpe_difference_values_and_input_handling():
    rng = np.random.default_rng(6)
    r = rng.normal(0.0, 0.01, 300)
    lab = np.tile([1.0, 0.0, np.nan], 100)
    out = regime_sharpe_difference(r, lab, n_boot=200, seed=1)
    a, b = r[lab == 1.0], r[lab == 0.0]
    assert out["sharpe_true"] == pytest.approx(a.mean() / a.std(ddof=1) * np.sqrt(252))
    assert out["sharpe_false"] == pytest.approx(b.mean() / b.std(ddof=1) * np.sqrt(252))
    assert out["difference"] == pytest.approx(out["sharpe_true"] - out["sharpe_false"])
    assert out["n_true"] == 100 and out["n_false"] == 100  # NaN labels are dropped
    with pytest.raises(ValueError):
        regime_sharpe_difference(r, np.ones(300))


def test_trend_regime_uses_its_own_window_and_dispersion_is_smoothed():
    n = 400
    idx = pd.bdate_range("2016-01-01", periods=n)
    # up for 200 days, then a slide: the 21-day return turns negative long before the 126-day one
    mkt = pd.Series(np.r_[np.full(200, 0.001), np.full(200, -0.002)], index=idx)
    disp = pd.Series(0.01, index=idx)
    disp.iloc[150] = 0.06  # one spike in cross-sectional dispersion
    r = causal_regimes(mkt, disp, window=21, trend_window=126, min_periods=100)
    t = 200 + 40  # 40 days into the slide: trailing 126-day compounded return
    trailing = float(np.prod(1 + mkt.iloc[t - 126 : t]) - 1)  # through t - 1, the label's last day
    assert r["trend_up"].iloc[t] == float(trailing > 0)
    assert trailing > 0 and r["trend_up"].iloc[t] == 1.0  # still up on the long window ...
    short = float(np.prod(1 + mkt.iloc[t - 21 : t]) - 1)
    assert short < 0  # ... although the 21-day window (the wrong one) would say down
    assert r["trend_up"].iloc[t + 90] == 0.0  # and down once the slide dominates the window
    spike_days = r["disp_high"].iloc[151:171]  # the spike stays in the 21-day mean for three weeks
    assert spike_days.mean() > 0.9 and r["disp_high"].iloc[175:190].mean() < 0.5
