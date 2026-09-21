"""Residual signals: causality (the whole point), the OU score against statsmodels, hand-checked
reversal score, NaN handling, and the position state machine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from statarb.data.leakage import assert_no_lookahead
from statarb.signals.residuals import (
    ResidualConfig,
    _ou_score,
    residual_signals,
    states_from_scores,
)

CFG = dict(fit_window=120, refit_every=10, beta_window=40)
FIRST = 150


def frame(t=260, n=24, k=2, seed=0, idio=0.012):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, (t, k)) * np.array([0.01, 0.006, 0.004][:k])
    b = rng.normal(0, 1, (n, k))
    b[:, 0] = np.abs(b[:, 0]) + 0.5
    idx = pd.bdate_range("2020-01-01", periods=t)
    return pd.DataFrame(
        f @ b.T + rng.normal(0, idio, (t, n)), index=idx, columns=[f"S{i}" for i in range(n)]
    )


def cfg(**kw):
    return ResidualConfig(**{**CFG, "n_factors": 3, **kw})


# ---- causality --------------------------------------------------------------------------------
@pytest.mark.parametrize("signal", ["sscore", "reversal"])
@pytest.mark.parametrize("k", [0, 3])
def test_scores_sigma_and_betas_use_only_data_up_to_the_decision_day(signal, k):
    r = frame()
    c = cfg(signal=signal, n_factors=k)

    def fn(x: pd.DataFrame) -> pd.DataFrame:
        s = residual_signals(x, FIRST, c)
        cols = [pd.DataFrame(s.score, index=s.dates), pd.DataFrame(s.sigma_e, index=s.dates)]
        cols += [pd.DataFrame(s.beta[:, :, j], index=s.dates) for j in range(k)]
        cols.append(pd.DataFrame(s.eps, index=s.dates))
        out = pd.concat(cols, axis=1)
        out.columns = range(out.shape[1])
        return out

    cuts = [r.index[i] for i in (FIRST, FIRST + 3, FIRST + 10, FIRST + 55, FIRST + 100)]
    assert_no_lookahead(fn, r, cuts)


def test_the_leak_detector_would_catch_a_fit_that_peeks_at_the_decision_day():
    """Negative control: fitting the PCA on rows through t (not t-1) must change the past scores."""
    r = frame()
    base = residual_signals(r, FIRST, cfg())
    # the last fit row of segment 1 is row FIRST+9; perturbing it must change scores from FIRST+10 on
    p = r.copy()
    p.iloc[FIRST + 9] += 0.05 * np.random.default_rng(1).normal(size=r.shape[1])
    moved = residual_signals(p, FIRST, cfg())
    d = 10  # first decision day of the next segment uses row FIRST+9 in its PCA fit
    assert np.nanmax(np.abs(moved.beta[d + 10 :] - base.beta[d + 10 :])) > 1e-4
    # ... and the segment containing that row is unaffected before the row happens
    assert np.array_equal(moved.score[:9], base.score[:9], equal_nan=True)


@pytest.mark.parametrize("offset", [37, 30, 0])  # 30 and 0 are the first days of a refit segment
def test_betas_and_sigma_never_see_the_scored_day_but_the_residual_does(offset):
    r = frame()
    base = residual_signals(r, FIRST, cfg(signal="reversal"))
    t = FIRST + offset
    p = r.copy()
    p.iloc[t] += 0.03
    moved = residual_signals(p, FIRST, cfg(signal="reversal"))
    d = t - FIRST
    assert np.array_equal(moved.beta[d], base.beta[d])  # betas fitted through t - 1 only
    assert np.array_equal(moved.sigma_e[d], base.sigma_e[d])
    assert not np.allclose(moved.eps[d], base.eps[d])  # the out-of-sample residual moves
    assert np.all(np.abs(moved.eps[d] - base.eps[d]) < 0.3)


def test_residual_volatility_is_an_unbiased_estimate_of_the_idiosyncratic_volatility():
    """Divides by (window - k): with 40 days and 2 factors the naive divisor is 5 % too small."""
    rng = np.random.default_rng(11)
    t, n = 900, 60
    f = rng.normal(0, [0.010, 0.006], (t, 2))
    b = rng.normal(0, 1, (n, 2))
    b[:, 0] = np.abs(b[:, 0]) + 0.5
    r = pd.DataFrame(
        f @ b.T + rng.normal(0, 0.008, (t, n)), index=pd.bdate_range("2018-01-01", periods=t)
    )
    s = residual_signals(r, 520, ResidualConfig(n_factors=2, beta_window=40, signal="reversal"))
    assert np.nanmean(s.sigma_e**2) / 0.008**2 == pytest.approx(1.0, abs=0.025)


def test_history_requirement():
    with pytest.raises(ValueError, match="history"):
        residual_signals(frame(), 100, cfg())


# ---- the OU score against an independent implementation ---------------------------------------
def test_ou_score_matches_statsmodels_ols_and_the_closed_forms():
    rng = np.random.default_rng(3)
    w, n = 60, 6
    x = np.zeros((w + 1, n))
    b_true = np.array([0.9, 0.85, 0.8, 0.7, 0.6, 0.5])
    for j in range(1, w + 1):
        x[j] = b_true * x[j - 1] + rng.normal(0, 0.01, n)
    s = _ou_score(x, w, kappa_min=0.1)
    for i in range(n):
        x0, x1 = x[: w - 1, i], x[1:w, i]
        res = sm.OLS(x1, sm.add_constant(x0)).fit()
        a, b = res.params
        assert 0 < b < 1
        s2 = res.ssr / (w - 3)
        m, s_eq = a / (1 - b), np.sqrt(s2 / (1 - b**2))
        assert s[i] == pytest.approx((x[w, i] - m) / s_eq, rel=1e-9)


def test_ou_score_drops_random_walks_alternating_paths_and_slow_reversion():
    rng = np.random.default_rng(5)
    w = 60
    walk = np.cumsum(rng.normal(0, 0.01, w + 1))
    alt = np.array([(-0.9) ** j for j in range(w + 1)]) * 0.01  # b < 0
    fast = np.zeros(w + 1)
    for j in range(1, w + 1):
        fast[j] = 0.6 * fast[j - 1] + rng.normal(0, 0.01)
    x = np.column_stack([walk, alt, fast])
    s = _ou_score(x, w, kappa_min=8.4)
    assert np.isnan(s[1]) and np.isfinite(s[2])  # kappa(0.6) = 128 > 8.4
    slow = np.zeros(w + 1)
    for j in range(1, w + 1):
        slow[j] = 0.97 * slow[j - 1] + rng.normal(0, 0.01)  # kappa = 7.7 < 8.4 on average
    assert np.isnan(_ou_score(np.column_stack([slow]), w, kappa_min=25.0)[0])


def test_sscore_sign_positive_when_the_cumulative_residual_is_above_its_mean():
    rng = np.random.default_rng(6)
    w = 60
    x = np.zeros((w + 1, 2))
    for j in range(1, w + 1):
        x[j] = 0.7 * x[j - 1] + rng.normal(0, 0.01, 2)
    x[w] = [x[:w, 0].mean() + 0.08, x[:w, 1].mean() - 0.08]
    s = _ou_score(x, w, kappa_min=0.1)
    assert s[0] > 2 and s[1] < -2


# ---- the reversal score, by hand --------------------------------------------------------------
def test_reversal_score_without_a_factor_model_matches_a_direct_computation():
    r = frame(k=1)
    c = cfg(signal="reversal", n_factors=0, reversal_days=5)
    s = residual_signals(r, FIRST, c)
    x = r.to_numpy()
    for d in (0, 17, 60):
        t = FIRST + d
        win = x[t - 40 : t]
        sig = np.sqrt((win**2).sum(0) / 40)
        expect = (x[t] + win[-4:].sum(0)) / (sig * np.sqrt(5))
        assert np.allclose(s.score[d], expect, atol=1e-12)
        assert np.allclose(s.sigma_e[d], sig, atol=1e-12)
        assert np.allclose(s.eps[d], x[t])


def test_the_first_residual_of_a_planted_factor_model_is_close_to_the_idiosyncratic_shock():
    rng = np.random.default_rng(8)
    t, n, k = 700, 50, 2
    f = rng.normal(0, [0.01, 0.006], (t, k))
    b = rng.normal(0, 1, (n, k))
    b[:, 0] = np.abs(b[:, 0]) + 0.5
    idio = rng.normal(0, 0.008, (t, n))
    r = pd.DataFrame(f @ b.T + idio, index=pd.bdate_range("2019-01-01", periods=t))
    s = residual_signals(r, 520, ResidualConfig(n_factors=2, signal="reversal"))
    truth = idio[520:]
    corr = np.corrcoef(s.eps.ravel(), truth.ravel())[0, 1]
    assert corr > 0.95
    raw = residual_signals(r, 520, ResidualConfig(n_factors=0, signal="reversal"))
    assert np.corrcoef(raw.eps.ravel(), truth.ravel())[0, 1] < corr - 0.15


# ---- missing data -----------------------------------------------------------------------------
def test_nan_in_the_window_or_on_the_day_removes_only_that_stock():
    r = frame()
    base = residual_signals(r, FIRST, cfg(signal="reversal"))
    p = r.copy()
    p.iloc[FIRST + 30, 4] = np.nan
    s = residual_signals(p, FIRST, cfg(signal="reversal"))
    d = 30
    assert np.isnan(s.score[d, 4])
    assert np.isnan(s.score[d : d + 41, 4]).all()  # leaves the regression window after 40 days
    assert np.isfinite(s.score[d + 41, 4])
    other = np.delete(np.arange(24), 4)
    # other stocks' scores change only through the (fixed) factor space -- not through a NaN
    assert np.isfinite(s.score[d][other]).all()
    assert np.isfinite(base.score[d]).all()


def test_a_nan_in_the_fit_window_drops_the_stock_from_the_pca_only():
    r = frame()
    p = r.copy()
    p.iloc[FIRST - 50, 7] = np.nan
    s = residual_signals(p, FIRST, cfg(signal="reversal"))
    assert s.fits[0]["n_fit"] == 23
    assert np.isfinite(s.score[0, 7])  # still scored: its own regression window is clean
    assert s.fits[-1]["n_fit"] == 24  # the gap has left the fit window


def test_zero_factor_control_records_no_fits_and_has_empty_betas():
    s = residual_signals(frame(), FIRST, cfg(n_factors=0, signal="reversal"))
    assert s.fits == [] and s.beta.shape[2] == 0


# ---- the position state machine ---------------------------------------------------------------
def col(*x):
    return np.array(x, dtype=float)[:, None]


def run(x, **kw):
    return states_from_scores(np.vstack([col(*x), col(0.0)]), **kw)[:-1, 0].tolist()


def test_long_and_short_entries_exits_and_strict_thresholds():
    assert run([0, -1.3, -1.0, -0.6, -0.4, 0, -1.25, -1.3]) == [0, 1, 1, 1, 0, 0, 0, 1]
    assert run([0, 1.3, 1.0, 0.6, 0.4, 0, 1.25, 1.4]) == [0, -1, -1, -1, 0, 0, 0, -1]


def test_stop_blocks_reentry_until_the_score_returns_inside_the_exit_band():
    x = [-1.5, -4.5, -1.5, -1.5, -0.2, -1.5]
    assert run(x) == [1, 0, 0, 0, 0, 1]
    assert run([-1.5, -4.5, -4.2]) == [1, 0, 0]  # beyond the stop nothing is entered


def test_time_stop_exits_and_blocks_reentry():
    x = [-1.5] * 8 + [0.0, -1.5]
    assert run(x, max_hold=3) == [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]


def test_nan_flattens_without_blocking():
    assert run([-1.5, -1.5, np.nan, -1.5, -1.5]) == [1, 1, 0, 1, 1]


def test_last_row_is_flat_and_states_are_causal_and_independent_across_stocks():
    rng = np.random.default_rng(9)
    s = rng.normal(0, 1.6, (300, 12))
    s[rng.random(s.shape) < 0.05] = np.nan
    full = states_from_scores(s)
    assert (full[-1] == 0).all()
    for cut in (40, 150):
        assert np.array_equal(states_from_scores(s[:cut])[:-1], full[: cut - 1])  # prefix property
    for j in (0, 5, 11):
        assert np.array_equal(states_from_scores(s[:, [j]])[:, 0], full[:, j])


def test_a_fresh_entry_is_not_taken_beyond_the_stop_and_nan_flattens_even_without_an_exit_band():
    assert run([0, -4.5, -1.5]) == [0, 0, 1]  # beyond the stop from flat: not entered (not blocked)
    assert run([0, 4.5, 1.5]) == [0, 0, -1]
    assert run([-1.5, np.nan, -1.5], exit=0.0) == [1, 0, 1]  # NaN flattens; it is not "score 0"
