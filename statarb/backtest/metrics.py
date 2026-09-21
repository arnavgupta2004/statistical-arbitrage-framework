"""Performance metrics for a daily return series.

All ratios are annualised with 252 periods and assume a zero risk-free rate (the series are
dollar-neutral long/short returns on capital, so excess-over-cash is a second-order correction that
Stage 6 revisits together with financing).  Nothing here is estimated with hindsight: the functions
are pure summaries of the series they are given.

* ``sortino``   mean / downside deviation (root mean square of returns below 0).
* ``max_drawdown``  worst peak-to-trough decline of the cumulative-return path (simple compounding).
* ``drawdown_duration``  longest stretch below a previous peak, in periods.
* ``calmar``    annualised mean return / |max drawdown|.
* ``profit_factor``  gross gains / gross losses.
* ``hit_rate``  share of periods with a positive return (flat periods excluded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = 252


def _arr(r) -> np.ndarray:
    return np.asarray(r, dtype=float)


def sharpe(r, periods: int = PERIODS) -> float:
    r = _arr(r)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def sortino(r, periods: int = PERIODS) -> float:
    r = _arr(r)
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return float(r.mean() / downside * np.sqrt(periods)) if downside > 0 else float("nan")


def equity_curve(r) -> np.ndarray:
    return np.cumprod(1.0 + _arr(r))


def drawdowns(r) -> np.ndarray:
    eq = equity_curve(r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return eq / peak - 1.0


def max_drawdown(r) -> float:
    dd = drawdowns(r)
    return float(dd.min()) if len(dd) else float("nan")


def drawdown_duration(r) -> int:
    """Longest run of consecutive periods spent below the running peak."""
    below = drawdowns(r) < 0
    best = run = 0
    for b in below:
        run = run + 1 if b else 0
        best = max(best, run)
    return int(best)


def calmar(r, periods: int = PERIODS) -> float:
    mdd = max_drawdown(r)
    return float(_arr(r).mean() * periods / abs(mdd)) if mdd < 0 else float("nan")


def profit_factor(r) -> float:
    r = _arr(r)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else float("inf") if gains > 0 else float("nan")


def hit_rate(r) -> float:
    r = _arr(r)
    active = r[r != 0]
    return float((active > 0).mean()) if len(active) else float("nan")


def performance_summary(returns, turnover=None, periods: int = PERIODS) -> dict:
    """Headline statistics of a daily return series (and its turnover, if given)."""
    r = _arr(returns)
    out = {
        "n_days": int(len(r)),
        "mean_daily_bps": float(r.mean() * 1e4),
        "ann_return_pct": float(r.mean() * periods * 100),
        "ann_vol_pct": float(r.std(ddof=1) * np.sqrt(periods) * 100)
        if len(r) > 1
        else float("nan"),
        "sharpe": sharpe(r, periods),
        "sortino": sortino(r, periods),
        "max_drawdown_pct": max_drawdown(r) * 100,
        "drawdown_duration_days": drawdown_duration(r),
        "calmar": calmar(r, periods),
        "hit_rate": hit_rate(r),
        "profit_factor": profit_factor(r),
    }
    if turnover is not None:
        out["turnover_per_year"] = float(_arr(turnover).sum() / (len(r) / periods))
    return out


def by_year(returns: pd.Series) -> pd.DataFrame:
    """Per-calendar-year summary (needs a DatetimeIndex)."""
    rows = []
    for year, g in returns.groupby(returns.index.year):
        rows.append(
            {
                "year": int(year),
                **{
                    k: v
                    for k, v in performance_summary(g.to_numpy()).items()
                    if k in ("n_days", "mean_daily_bps", "sharpe", "max_drawdown_pct")
                },
            }
        )
    return pd.DataFrame(rows)


def rolling_sharpe(returns: pd.Series, window: int = 126, periods: int = PERIODS) -> pd.Series:
    """Annualised Sharpe over the trailing ``window`` days (causal: row ``t`` uses ``<= t``)."""
    roll = returns.rolling(window)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = roll.mean() / roll.std(ddof=1) * np.sqrt(periods)
    return out.where(np.isfinite(out))


def rolling_volatility(returns: pd.Series, window: int = 63, periods: int = PERIODS) -> pd.Series:
    """Annualised volatility over the trailing ``window`` days (causal)."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods)
