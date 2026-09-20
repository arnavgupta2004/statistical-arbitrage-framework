"""Stationary (block) bootstrap for dependent series -- Politis & Romano (1994).

Daily strategy returns are serially dependent (positions persist, volatility clusters), so an iid
bootstrap understates the uncertainty of a Sharpe ratio.  Resampling *blocks* of random geometric
length (mean ``mean_block``) keeps the dependence structure while keeping the resampled series
stationary.  Indices wrap around the sample (circular), which is what makes the bootstrap
stationary.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One resample of ``range(n)``: blocks start anywhere and end w.p. ``1/mean_block``."""
    if n < 2 or mean_block < 1:
        raise ValueError("need n >= 2 and mean_block >= 1")
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    restart = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if restart[t] else (idx[t - 1] + 1) % n
    return idx


def bootstrap_ci(
    x,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    mean_block: float = 10.0,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, tuple[float, float], np.ndarray]:
    """Point estimate, percentile interval and the bootstrap distribution of ``statistic(x)``."""
    xv = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            statistic(xv[stationary_bootstrap_indices(len(xv), mean_block, rng)])
            for _ in range(n_boot)
        ]
    )
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(statistic(xv)), (float(lo), float(hi)), draws


def sharpe(r: np.ndarray, periods: int = 252) -> float:
    """Annualised Sharpe of a daily return series (zero risk-free); NaN if the series is flat."""
    sd = np.std(r, ddof=1)
    return float(np.mean(r) / sd * np.sqrt(periods)) if sd > 0 else float("nan")
