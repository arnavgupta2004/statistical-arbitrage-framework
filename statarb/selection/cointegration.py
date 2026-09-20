"""Engle-Granger two-step cointegration test.

Model and hypotheses
--------------------
Step 1 -- estimate the long-run relation by OLS on the *training* data::

    Y_t = alpha + beta' X_t + e_t     ->  hedge ratio beta; spread S_t = Y_t - beta' X_t (- alpha)

Step 2 -- test the residual spread for a unit root with an ADF regression (no deterministic terms,
because the residuals have mean zero by construction):

* H0: the residual has a unit root -- Y and X are **not** cointegrated (any linear combination
  drifts; the regression is spurious).
* H1: the residual is stationary -- Y and X are cointegrated with cointegrating vector (1, -beta).

Why not the ordinary ADF table
------------------------------
The residual is not observed: OLS chose ``beta`` to make it look as stationary as possible.  The
null distribution of the statistic is therefore shifted left of the Dickey-Fuller distribution, and
using standard ADF critical values would reject far too often (``tests/test_cointegration.py`` shows
this).  p-values and critical values come from MacKinnon's cointegration surfaces, which depend
on the number of variables in the regression and on the first-stage deterministic terms.

Caveats
-------
* **Asymmetry.**  Regressing Y on X and X on Y give different statistics in finite samples.  Testing
  both and keeping the better one is data-snooping; ``engle_granger_both`` returns both and states
  ``n_tests = 2`` so that selection can be corrected.
* Statistical cointegration in-sample is neither stable out of sample nor sufficient for profit.
* The hedge ratio is a random variable; its sampling error is not a nuisance for the strategy but a
  cost (Stage 4 studies hedge-ratio stability).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp

from statarb.selection.adf import ADFResult, adf_test, ols, trend_matrix


@dataclass(frozen=True)
class EngleGrangerResult:
    statistic: float
    pvalue: float
    critical_values: dict[str, float]
    hedge_ratio: np.ndarray  # beta, one entry per regressor in X
    intercept: float  # alpha (0.0 when trend="n")
    spread: pd.Series | np.ndarray  # Y - alpha - beta'X (the regression residual)
    adf: ADFResult
    rsquared: float
    nobs: int
    trend: str
    n_tests: int = 1

    def rejects_no_cointegration(self, alpha: float = 0.05) -> bool:
        return self.pvalue < alpha


def engle_granger(
    y,
    x,
    trend: str = "c",
    maxlag: int | None = None,
    autolag: str | None = "aic",
) -> EngleGrangerResult:
    """Engle-Granger test of H0 "no cointegration" for ``y`` against one or more series ``x``."""
    index = y.index if isinstance(y, pd.Series) else None
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if y_arr.ndim != 1 or len(y_arr) != len(x_arr):
        raise ValueError("y must be 1-d and x must have the same number of rows")
    if not (np.all(np.isfinite(y_arr)) and np.all(np.isfinite(x_arr))):
        raise ValueError("inputs contain NaN/inf; align and drop missing rows first")
    if trend not in ("n", "c", "ct", "ctt"):
        raise ValueError("trend must be one of 'n', 'c', 'ct', 'ctt'")

    n, k = x_arr.shape
    design = np.column_stack([x_arr, trend_matrix(n, trend)])
    beta_full, resid, ssr, _ = ols(y_arr, design)
    tss = float(((y_arr - y_arr.mean()) ** 2).sum())
    rsquared = 1.0 - ssr / tss
    if rsquared > 1 - 1e-8:
        raise ValueError("y and x are (almost) perfectly collinear; the test is not meaningful")

    adf = adf_test(resid, regression="n", maxlag=maxlag, autolag=autolag)
    n_vars = k + 1
    crit = mackinnoncrit(N=n_vars, regression=trend, nobs=n - 1) if trend != "n" else [np.nan] * 3
    return EngleGrangerResult(
        statistic=adf.statistic,
        pvalue=float(mackinnonp(adf.statistic, regression=trend, N=n_vars)),
        critical_values={"1%": crit[0], "5%": crit[1], "10%": crit[2]},
        hedge_ratio=beta_full[:k],
        intercept=float(beta_full[k]) if trend != "n" else 0.0,
        spread=pd.Series(resid, index=index) if index is not None else resid,
        adf=adf,
        rsquared=float(rsquared),
        nobs=n,
        trend=trend,
    )


def engle_granger_both(
    y, x, trend: str = "c", maxlag: int | None = None, autolag: str | None = "aic"
) -> tuple[EngleGrangerResult, EngleGrangerResult]:
    """Both regression directions (Y on X, X on Y).  Together they are **two** hypothesis tests."""
    fwd = engle_granger(y, x, trend, maxlag, autolag)
    rev = engle_granger(x, y, trend, maxlag, autolag)
    return (
        EngleGrangerResult(**{**fwd.__dict__, "n_tests": 2}),
        EngleGrangerResult(**{**rev.__dict__, "n_tests": 2}),
    )


def spread(y, x, hedge_ratio, intercept: float = 0.0):
    """``S_t = Y_t - intercept - beta' X_t`` for given (training-period) coefficients."""
    beta = np.atleast_1d(np.asarray(hedge_ratio, dtype=float))
    x_arr = np.asarray(x, dtype=float)
    x_arr = x_arr[:, None] if x_arr.ndim == 1 else x_arr
    out = np.asarray(y, dtype=float) - intercept - x_arr @ beta
    return pd.Series(out, index=y.index) if isinstance(y, pd.Series) else out
