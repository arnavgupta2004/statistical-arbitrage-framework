"""Causal market regimes, and a test of whether performance differs between them.

A regime label for the return of day ``t`` must be known at the close of ``t - 1``: every indicator
here uses data through ``t - 1`` and is compared with a threshold that is itself causal (the
*expanding* median of the indicator's own history, never the full-sample median).  Regimes are fixed
in advance:

* ``vol``    realised volatility of the equal-weight market over the last ``window`` days above its
             expanding median (stress);
* ``trend``  cumulative market return over the last ``trend_window`` days above zero (bull / bear);
* ``disp``   cross-sectional dispersion of daily stock returns, averaged over ``window`` days, above
             its expanding median (how much idiosyncratic movement there is to harvest).

``regime_sharpe_difference`` compares a strategy's Sharpe ratio in the two states of one regime
with a stationary bootstrap over dates (labels travel with their dates) and returns a two-sided
p-value for "the Sharpe ratios are equal" from the centred bootstrap distribution of the difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.statistics.bootstrap import stationary_bootstrap_matrix


def expanding_above_median(x: pd.Series, min_periods: int = 252) -> pd.Series:
    """1.0 where ``x_t`` beats the median of ``x_s, s <= t``; NaN before ``min_periods``."""
    med = x.expanding(min_periods=min_periods).median()
    return (x > med).where(med.notna()).astype(float)


def causal_regimes(
    market_ret: pd.Series,
    dispersion: pd.Series,
    window: int = 21,
    trend_window: int = 126,
    min_periods: int = 252,
) -> pd.DataFrame:
    """Regime table (1.0 / 0.0 / NaN) like ``market_ret``; row ``t`` uses data through ``t - 1``."""
    vol = market_ret.rolling(window).std()
    disp = dispersion.rolling(window).mean()
    trend = (1 + market_ret).rolling(trend_window).apply(np.prod, raw=True) - 1
    out = pd.DataFrame(
        {
            "vol_high": expanding_above_median(vol, min_periods),
            "trend_up": (trend > 0).where(trend.notna()).astype(float),
            "disp_high": expanding_above_median(disp, min_periods),
        }
    )
    return out.shift(1)  # the label for day t uses information through t - 1


def _sharpe_rows(r: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-row Sharpe (unannualised) of ``r`` restricted to ``mask`` (both B x n)."""
    cnt = mask.sum(axis=1)
    s1 = np.where(mask, r, 0.0).sum(axis=1)
    s2 = np.where(mask, r**2, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var = (s2 - cnt * mean**2) / (cnt - 1)
        return np.where((cnt > 2) & (var > 0), mean / np.sqrt(var), np.nan)


def regime_sharpe_difference(
    r, labels, n_boot: int = 2000, mean_block: float = 10.0, seed: int = 0, periods: int = 252
) -> dict:
    """Annualised Sharpe in the ``True`` / ``False`` states, their difference, bootstrap p-value."""
    r = np.asarray(r, dtype=float)
    lab = np.asarray(labels, dtype=float)
    keep = np.isfinite(lab) & np.isfinite(r)
    r, lab = r[keep], lab[keep].astype(bool)
    if lab.all() or (~lab).all() or len(r) < 40:
        raise ValueError("both regime states need observations")
    ann = np.sqrt(periods)
    obs_t = _sharpe_rows(r[None, :], lab[None, :])[0] * ann
    obs_f = _sharpe_rows(r[None, :], ~lab[None, :])[0] * ann
    idx = stationary_bootstrap_matrix(len(r), n_boot, mean_block, np.random.default_rng(seed))
    rb, lb = r[idx], lab[idx]
    diff_b = (_sharpe_rows(rb, lb) - _sharpe_rows(rb, ~lb)) * ann
    diff = obs_t - obs_f
    valid = diff_b[np.isfinite(diff_b)]
    p = float((1 + np.sum(np.abs(valid - diff) >= abs(diff))) / (len(valid) + 1))
    return {
        "sharpe_true": float(obs_t),
        "sharpe_false": float(obs_f),
        "difference": float(diff),
        "p_value": p,
        "n_true": int(lab.sum()),
        "n_false": int((~lab).sum()),
    }
