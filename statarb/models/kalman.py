"""Kalman-filter time-varying hedge ratio.

A stretch goal: compared against the simpler methods, never assumed better.

Model (state ``[beta_t, a_t]`` follows a random walk; observation is the log price of Y)::

    ly_t = beta_t * lx_c,t + a_t + e_t,      e_t ~ N(0, R)
    [beta_t, a_t] = [beta_{t-1}, a_{t-1}] + w_t,     w_t ~ N(0, Vw),   Vw = delta/(1-delta) * I

``lx_c`` is ``lx`` minus its mean over the first ``n_init`` observations.  Centring decouples the
intercept from the slope (log prices sit near 4, which would otherwise make ``a`` and ``beta``
almost collinear); the reported intercept is converted back to the uncentred convention.

Causality: the prior is the OLS fit on the first ``n_init`` observations, with its own
covariance; the observation variance ``R`` is the OLS residual variance there.  Everything at
``t >= n_init`` is then a pure forward recursion, so the output at ``t`` uses data ``<= t`` only.
Output before ``n_init`` is NaN.

``delta`` is the knob: the state noise per day.  Its useful range depends on how fast the true hedge
ratio moves relative to the observation noise; it is not "the" right value, so the experiments sweep
it.  As ``delta -> 0`` the filter converges to recursive least squares (expanding OLS), which the
tests use as an analytic check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_hedge(y, x, delta: float = 1e-4, n_init: int = 60) -> pd.DataFrame:
    """Filtered ``beta``, ``alpha`` and the one-step innovation for ``ly ~ alpha + beta * lx``."""
    index = y.index if isinstance(y, pd.Series) else None
    yv, xv = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    n = len(yv)
    if not (np.all(np.isfinite(yv)) and np.all(np.isfinite(xv))):
        raise ValueError("inputs contain NaN/inf; the filter needs gap-free series")
    if n_init < 10 or n <= n_init:
        raise ValueError("need n_init >= 10 and more observations than n_init")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")

    x_mean = xv[:n_init].mean()
    xc = xv - x_mean
    design = np.column_stack([xc[:n_init], np.ones(n_init)])
    coef, *_ = np.linalg.lstsq(design, yv[:n_init], rcond=None)
    resid = yv[:n_init] - design @ coef
    r_obs = float(resid @ resid / (n_init - 2))
    state = coef.copy()  # [beta, a_centred]
    cov = r_obs * np.linalg.inv(design.T @ design)
    q = delta / (1.0 - delta)

    beta = np.full(n, np.nan)
    alpha = np.full(n, np.nan)
    innov = np.full(n, np.nan)
    innov_sd = np.full(n, np.nan)
    for t in range(n_init, n):
        cov_pred = cov + q * np.eye(2)
        h0, h1 = xc[t], 1.0
        ph0 = cov_pred[0, 0] * h0 + cov_pred[0, 1] * h1
        ph1 = cov_pred[1, 0] * h0 + cov_pred[1, 1] * h1
        s = h0 * ph0 + h1 * ph1 + r_obs
        nu = yv[t] - (h0 * state[0] + h1 * state[1])
        k0, k1 = ph0 / s, ph1 / s
        state = state + np.array([k0, k1]) * nu
        # Joseph-free but symmetric update: P = P_pred - K S K'
        cov = cov_pred - s * np.outer([k0, k1], [k0, k1])
        cov = (cov + cov.T) / 2.0
        beta[t], alpha[t] = state[0], state[1] - state[0] * x_mean
        innov[t], innov_sd[t] = nu, np.sqrt(s)
    out = pd.DataFrame(
        {"alpha": alpha, "beta": beta, "innovation": innov, "innovation_sd": innov_sd}
    )
    if index is not None:
        out.index = index
    return out
