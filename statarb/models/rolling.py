"""Causal rolling statistics.

Every function returns, at row ``t``, a value computed from rows ``<= t`` only (a trailing window
that *includes* the current observation, which is information available at the close of ``t``).
``exclude_current=True`` drops the current row from the window, for signals that must not
standardise a value by a statistic that already contains it.  All of these pass the look-ahead
detectors in ``statarb.data.leakage`` (see ``tests/test_rolling_hurst.py``).
"""

from __future__ import annotations

import pandas as pd


def _window(x: pd.Series | pd.DataFrame, window: int | None, min_periods: int | None):
    if window is None:
        return x.expanding(min_periods=min_periods or 2)
    return x.rolling(window, min_periods=min_periods or window)


def rolling_mean_std(
    x,
    window: int | None,
    min_periods: int | None = None,
    ddof: int = 1,
    exclude_current: bool = False,
):
    """Trailing mean and standard deviation; ``window=None`` gives an expanding window."""
    roller = _window(x.shift(1) if exclude_current else x, window, min_periods)
    return roller.mean(), roller.std(ddof=ddof)


def zscore(
    x,
    window: int | None,
    min_periods: int | None = None,
    ddof: int = 1,
    exclude_current: bool = False,
):
    """``(x_t - mean_t) / std_t``; a zero-variance window gives NaN, never +/-inf."""
    mean, std = rolling_mean_std(x, window, min_periods, ddof, exclude_current)
    return (x - mean) / std.where(std > 0)


def rolling_beta(y: pd.Series, x: pd.Series, window: int | None, min_periods: int | None = None):
    """Trailing OLS slope and intercept of ``y`` on ``x`` (``window=None``: expanding)."""
    my = _window(y, window, min_periods).mean()
    mx = _window(x, window, min_periods).mean()
    cov = _window(y * x, window, min_periods).mean() - my * mx
    var = _window(x * x, window, min_periods).mean() - mx * mx
    beta = cov / var.where(var > 1e-14 * (mx.abs() + 1) ** 2)
    return beta, my - beta * mx


def ewm_mean_std(x, halflife: float, min_periods: int = 2):
    """Exponentially weighted mean and standard deviation (adjusted weights, trailing)."""
    e = x.ewm(halflife=halflife, min_periods=min_periods)
    return e.mean(), e.std()
