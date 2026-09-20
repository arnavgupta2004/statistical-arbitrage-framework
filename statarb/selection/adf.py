"""Augmented Dickey-Fuller unit-root test, implemented from first principles.

Hypotheses
----------
Fit ``dy_t = a + d*t + g*y_{t-1} + sum_i phi_i dy_{t-i} + e_t`` (deterministic terms per
``regression``) and test the coefficient on the lagged level:

* H0: ``g = 0``  -- the series has a unit root (is non-stationary, I(1)).
* H1: ``g < 0``  -- the series is stationary around the chosen deterministic component.

The statistic is the OLS t-ratio of ``g``.  It does **not** follow a Student-t law under H0;
its null distribution is the Dickey-Fuller distribution, so p-values and critical values come
from MacKinnon's response surfaces (``statsmodels.tsa.stattools.mackinnonp / mackinnoncrit``,
published coefficient tables that are used here as data, not re-derived).

Assumptions and what they mean in practice
------------------------------------------
* The lag augmentation must whiten the errors.  Too few lags leaves serial correlation and
  distorts the size; too many costs power.  Lags are chosen by AIC/BIC on a common sample
  (Schwert's ``12*(n/100)^(1/4)`` rule for the upper bound), then the model is re-fitted on the
  longest sample available for the chosen lag.
* Errors are homoskedastic.  Volatility clustering, fat tails and structural breaks (all common
  in prices) are *not* handled; a break towards a new mean is read as a unit root, so the test
  is biased towards non-rejection when relationships shift.
* The deterministic specification must contain the true one.  ``regression="c"`` for a
  mean-reverting spread, ``"n"`` for regression residuals (their mean is already zero).

Interpretation
--------------
Rejecting H0 is *statistical* evidence of mean reversion in-sample.  It says nothing about
economic profitability (cost, capacity, speed of reversion) and, run across many candidate series,
it guarantees false positives: every call is one hypothesis test and must be counted when selection
is corrected for multiple testing (Stage 8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp

N_TREND = {"n": 0, "c": 1, "ct": 2, "ctt": 3}
MIN_OBS = 10  # below this the regression has too few degrees of freedom to mean anything


@dataclass(frozen=True)
class ADFResult:
    statistic: float
    pvalue: float
    usedlag: int
    nobs: int
    critical_values: dict[str, float]
    regression: str
    autolag: str | None
    icbest: float | None

    def rejects_unit_root(self, alpha: float = 0.05) -> bool:
        return self.pvalue < alpha


def trend_matrix(nobs: int, regression: str) -> np.ndarray:
    """Deterministic regressors ``[1, t, t^2]`` (as many as ``regression`` names)."""
    if regression not in N_TREND:
        raise ValueError(f"regression must be one of {sorted(N_TREND)}, got {regression!r}")
    t = np.arange(1, nobs + 1, dtype=float)
    cols = [np.ones(nobs), t, t**2][: N_TREND[regression]]
    return np.column_stack(cols) if cols else np.empty((nobs, 0))


def ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """OLS via QR.  Returns ``(beta, residuals, ssr, (X'X)^-1)``; raises on rank deficiency."""
    q, r = np.linalg.qr(x)
    if np.min(np.abs(np.diag(r))) < 1e-12 * max(1.0, np.max(np.abs(np.diag(r)))):
        raise ValueError("regressors are (numerically) collinear")
    beta = solve_triangular(r, q.T @ y)
    resid = y - x @ beta
    r_inv = solve_triangular(r, np.eye(r.shape[0]))
    return beta, resid, float(resid @ resid), r_inv @ r_inv.T


def information_criterion(kind: str, ssr: float, nobs: int, k: int) -> float:
    """Gaussian AIC/BIC (constants included so values match a full OLS log-likelihood)."""
    ll_term = nobs * (np.log(2 * np.pi) + np.log(ssr / nobs) + 1.0)
    return ll_term + (2.0 * k if kind == "aic" else k * np.log(nobs))


def _design(x: np.ndarray, xdiff: np.ndarray, lag: int, nobs: int) -> np.ndarray:
    """Columns ``[y_{t-1}, dy_{t-1}, ..., dy_{t-lag}]`` for the last ``nobs`` differences."""
    start = len(xdiff) - nobs
    cols = [x[start : start + nobs]]
    cols += [xdiff[start - i : start - i + nobs] for i in range(1, lag + 1)]
    return np.column_stack(cols)


def adf_test(
    x,
    regression: str = "c",
    maxlag: int | None = None,
    autolag: str | None = "aic",
) -> ADFResult:
    """ADF test of H0 "unit root".  ``x`` must be a finite 1-d series (no NaN: drop them first)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN/inf; the test needs a gap-free series")
    if x.max() == x.min():
        raise ValueError("x is constant")
    if autolag not in (None, "aic", "bic"):
        raise ValueError("autolag must be None, 'aic' or 'bic'")
    if len(x) < MIN_OBS:
        raise ValueError(f"sample too short: need at least {MIN_OBS} observations")
    n, ntrend = len(x), N_TREND.get(regression)
    if ntrend is None:
        raise ValueError(f"regression must be one of {sorted(N_TREND)}")

    limit = n // 2 - ntrend - 1
    if maxlag is None:
        maxlag = min(limit, int(np.ceil(12.0 * (n / 100.0) ** 0.25)))
        if maxlag < 0:
            raise ValueError("sample too short for this regression")
    elif maxlag > limit:
        raise ValueError(f"maxlag must not exceed n/2 - ntrend - 1 = {limit}")

    xdiff = np.diff(x)
    icbest = None
    if autolag is None:
        lag = maxlag
    else:
        nmax = len(xdiff) - maxlag  # common sample so the criteria are comparable across lags
        full = _design(x, xdiff, maxlag, nmax)
        trend = trend_matrix(nmax, regression)
        y = xdiff[maxlag:]
        scores = []
        for lag_i in range(maxlag + 1):
            xm = np.column_stack([full[:, : lag_i + 1], trend])
            _, _, ssr, _ = ols(y, xm)
            scores.append((information_criterion(autolag, ssr, nmax, xm.shape[1]), lag_i))
        icbest, lag = min(scores)  # ties resolve to the shortest lag

    nobs = len(xdiff) - lag
    xm = np.column_stack([_design(x, xdiff, lag, nobs), trend_matrix(nobs, regression)])
    beta, _, ssr, xtx_inv = ols(xdiff[lag:], xm)
    s2 = ssr / (nobs - xm.shape[1])
    stat = float(beta[0] / np.sqrt(s2 * xtx_inv[0, 0]))

    crit = mackinnoncrit(N=1, regression=regression, nobs=nobs)
    return ADFResult(
        statistic=stat,
        pvalue=float(mackinnonp(stat, regression=regression, N=1)),
        usedlag=lag,
        nobs=nobs,
        critical_values={"1%": crit[0], "5%": crit[1], "10%": crit[2]},
        regression=regression,
        autolag=autolag,
        icbest=None if icbest is None else float(icbest),
    )
