"""Hurst exponent estimators -- a *secondary, exploratory* diagnostic only.

H < 0.5 suggests mean reversion, 0.5 a random walk, > 0.5 persistence.  It is deliberately
**not** used as a selection criterion anywhere in this project.  Any use must keep in mind:

* It is an *estimator* with a large finite-sample variance, not a hypothesis test: there is no
  null distribution and no p-value here.  Two estimators can disagree by 0.1 on the same data.
* Rescaled-range (R/S) is biased upward in short samples (iid noise gives ~0.55-0.6, not 0.5), so
  "H > 0.5" is not evidence of trending unless compared with the estimator's own null.
* The variance-of-differences estimator measures the *scaling over the chosen lag range*; a
  stationary mean-reverting series only looks like ``H < 0.5`` at short lags and saturates at
  long ones, so the answer depends on the lag window.
* Structural breaks, heavy tails and volatility clustering all move it.

Stationarity (ADF/Engle-Granger) and the OU half-life are the decision tools; this is a sanity plot.
"""

from __future__ import annotations

import numpy as np


def hurst_variance(x, max_lag: int = 50, min_lag: int = 2) -> float:
    """Slope of ``log std(x_{t+tau} - x_t)`` on ``log tau``; use on a *level* series."""
    x = np.asarray(x, dtype=float)
    if len(x) < 4 * max_lag:
        raise ValueError("series too short for this max_lag")
    lags = np.arange(min_lag, max_lag + 1)
    tau_std = np.array([np.std(x[lag:] - x[:-lag], ddof=1) for lag in lags])
    return float(np.polyfit(np.log(lags), np.log(tau_std), 1)[0])


def hurst_rs(increments, min_chunk: int = 8) -> float:
    """Classical rescaled-range estimate on an *increment* series; biased upward (see module)."""
    r = np.asarray(increments, dtype=float)
    n = len(r)
    if n < 4 * min_chunk:
        raise ValueError("series too short")
    sizes = np.unique(np.floor(np.logspace(np.log10(min_chunk), np.log10(n // 2), 12)).astype(int))
    log_n, log_rs = [], []
    for s in sizes:
        chunks = r[: (n // s) * s].reshape(-1, s)
        dev = np.cumsum(chunks - chunks.mean(axis=1, keepdims=True), axis=1)
        rng = dev.max(axis=1) - dev.min(axis=1)
        sd = chunks.std(axis=1, ddof=0)
        ok = sd > 0
        if ok.any():
            log_n.append(np.log(s))
            log_rs.append(np.log((rng[ok] / sd[ok]).mean()))
    return float(np.polyfit(log_n, log_rs, 1)[0])
