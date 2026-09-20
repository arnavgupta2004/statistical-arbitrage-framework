"""Ornstein-Uhlenbeck model of a mean-reverting spread.

Continuous time::   dS = kappa (theta - S) dt + sigma dW

Exact discretisation at step ``dt`` (no Euler error) is an AR(1) with Gaussian innovations::

    S_{t+dt} = a + b S_t + e,     b = exp(-kappa dt),   a = theta (1 - b),
    Var(e)   = sigma^2 (1 - b^2) / (2 kappa)

so the parameters follow from an AR(1) fit::

    kappa = -ln(b)/dt,   theta = a/(1-b),   sigma^2 = 2 kappa Var(e) / (1 - b^2),
    half-life = ln 2 / kappa,   stationary sd = sigma / sqrt(2 kappa)

``0 < b < 1`` is required.  If the fitted ``b >= 1`` the sample shows no mean reversion and
``kappa`` is undefined (``mean_reverting=False``, fields NaN); if ``b <= 0`` the reversion is faster
than one step and the continuous-time reading is not meaningful.  Units: with ``dt = 1`` the
half-life is in *observation steps* (trading days for daily data).

Estimation error is the whole story for half-lives
-------------------------------------------------
* The OLS estimate of ``b`` is biased **downward** in finite samples (Kendall: about
  ``-(1 + 3b)/n``), hence ``kappa`` and the speed of reversion are biased **upward** -- a naive fit
  makes spreads look faster than they are.  ``bias_correct=True`` applies Kendall's correction.
* The half-life is a convex, unbounded function of ``b``: a confidence interval for ``b`` that
  reaches 1 gives an *infinite* upper bound.  ``half_life_ci`` maps the interval for ``b``
  endpoint-wise, so it is asymmetric and can be infinite -- which is the honest answer.
* ``method="mle"`` maximises the exact Gaussian likelihood *including* the stationary density of the
  first observation; with a few hundred points it barely differs from OLS, and it is provided
  mainly as a cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

LN2 = float(np.log(2.0))


@dataclass(frozen=True)
class OUFit:
    kappa: float
    theta: float
    sigma: float
    half_life: float
    b: float  # AR(1) coefficient exp(-kappa dt)
    a: float
    sigma_eps: float  # innovation sd of the discrete AR(1)
    b_se: float
    half_life_ci: tuple[float, float]
    n: int  # observations used
    dt: float
    mean_reverting: bool
    method: str

    @property
    def stationary_sd(self) -> float:
        return self.sigma / np.sqrt(2.0 * self.kappa) if self.mean_reverting else float("nan")


def half_life_from_b(b: float, dt: float = 1.0) -> float:
    """``ln 2 / kappa`` for AR(1) coefficient ``b`` (NaN unless ``0 < b < 1``)."""
    if not 0.0 < b < 1.0:
        return float("nan")
    return LN2 * dt / -np.log(b)


def _nan_fit(b, a, s_eps, b_se, n, dt, method) -> OUFit:
    nan = float("nan")
    return OUFit(nan, nan, nan, nan, b, a, s_eps, b_se, (nan, nan), n, dt, False, method)


def fit_ou(
    x,
    dt: float = 1.0,
    method: str = "ols",
    bias_correct: bool = False,
    ci_level: float = 0.95,
) -> OUFit:
    """Fit an OU process to the level series ``x`` (finite, equally spaced)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite 1-d array")
    if len(x) < 10:
        raise ValueError("need at least 10 observations")
    if method not in ("ols", "mle"):
        raise ValueError("method must be 'ols' or 'mle'")

    lag, lead = x[:-1], x[1:]
    n = len(lead)
    lag_c = lag - lag.mean()
    sxx = float(lag_c @ lag_c)
    b = float(lag_c @ (lead - lead.mean()) / sxx)
    a = float(lead.mean() - b * lag.mean())
    resid = lead - a - b * lag
    s2 = float(resid @ resid) / (n - 2)
    b_se = float(np.sqrt(s2 / sxx))
    if bias_correct:
        b = b + (1.0 + 3.0 * b) / (n + 1)  # Kendall (1954); n + 1 = number of levels
        a = float(lead.mean() - b * lag.mean())

    if method == "mle":
        return _fit_mle(x, dt, b, a, np.sqrt(s2), b_se, ci_level)
    return _from_ar1(b, a, np.sqrt(s2), b_se, n, dt, ci_level, "ols")


def _from_ar1(b, a, s_eps, b_se, n, dt, ci_level, method) -> OUFit:
    if not 0.0 < b < 1.0:
        return _nan_fit(b, a, s_eps, b_se, n, dt, method)
    kappa = -np.log(b) / dt
    z = stats.norm.ppf(0.5 + ci_level / 2.0)
    lo_b, hi_b = b - z * b_se, b + z * b_se
    hl_lo = half_life_from_b(lo_b, dt) if 0.0 < lo_b < 1.0 else (0.0 if lo_b <= 0.0 else np.inf)
    hl_hi = half_life_from_b(hi_b, dt) if 0.0 < hi_b < 1.0 else np.inf
    return OUFit(
        kappa=float(kappa),
        theta=float(a / (1.0 - b)),
        sigma=float(s_eps * np.sqrt(2.0 * kappa / (1.0 - b * b))),
        half_life=float(LN2 / kappa),
        b=float(b),
        a=float(a),
        sigma_eps=float(s_eps),
        b_se=float(b_se),
        half_life_ci=(float(hl_lo), float(hl_hi)),
        n=n,
        dt=dt,
        mean_reverting=True,
        method=method,
    )


def _neg_loglik(params, x, dt) -> float:
    log_kappa, theta, log_sigma = params
    kappa, sigma = np.exp(log_kappa), np.exp(log_sigma)
    b = np.exp(-kappa * dt)
    var_inf = sigma**2 / (2 * kappa)  # stationary variance
    var_step = var_inf * (1 - b * b)
    mean = theta + (x[:-1] - theta) * b
    ll = stats.norm.logpdf(x[0], theta, np.sqrt(var_inf)).sum()
    ll += stats.norm.logpdf(x[1:], mean, np.sqrt(var_step)).sum()
    return -float(ll)


def _fit_mle(x, dt, b0, a0, s_eps0, b_se, ci_level) -> OUFit:
    n = len(x) - 1
    if 0.0 < b0 < 1.0:
        kappa0, theta0 = -np.log(b0) / dt, a0 / (1.0 - b0)
        sigma0 = s_eps0 * np.sqrt(2 * kappa0 / (1 - b0 * b0))
    else:  # no reversion in the sample: start slow, the optimiser decides
        kappa0, theta0, sigma0 = 0.01 / dt, float(x.mean()), float(np.std(np.diff(x)) / np.sqrt(dt))
    res = optimize.minimize(
        _neg_loglik,
        [np.log(kappa0), theta0, np.log(sigma0)],
        args=(x, dt),
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 4000},
    )
    kappa, theta, sigma = np.exp(res.x[0]), res.x[1], np.exp(res.x[2])
    b = float(np.exp(-kappa * dt))
    s_eps = float(sigma * np.sqrt((1 - b * b) / (2 * kappa)))
    return _from_ar1(b, float(theta * (1 - b)), s_eps, b_se, n, dt, ci_level, "mle")


def simulate_ou(
    kappa: float,
    theta: float,
    sigma: float,
    n: int,
    dt: float = 1.0,
    x0: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Exact-discretisation OU path of ``n`` points; ``x0`` defaults to a stationary draw."""
    rng = rng or np.random.default_rng()
    b = np.exp(-kappa * dt)
    sd_step = sigma * np.sqrt((1 - b * b) / (2 * kappa))
    x = np.empty(n)
    x[0] = theta + sigma / np.sqrt(2 * kappa) * rng.standard_normal() if x0 is None else x0
    shocks = rng.standard_normal(n - 1) * sd_step
    for t in range(1, n):
        x[t] = theta + (x[t - 1] - theta) * b + shocks[t - 1]
    return x
