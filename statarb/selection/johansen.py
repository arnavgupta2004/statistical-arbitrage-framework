"""Johansen cointegration test for systems of ``n`` series (baskets), from first principles.

Model
-----
The VAR(k+1) in error-correction form::

    dy_t = Pi y_{t-1} + sum_{i=1..k} Gamma_i dy_{t-i} + deterministic + e_t,     Pi = alpha beta'

``rank(Pi) = r`` is the **cointegration rank**: ``r = 0`` means no long-run relation (n independent
stochastic trends), ``r = n`` means every series is stationary, and ``0 < r < n`` means ``r``
stationary linear combinations ``beta_i' y_t`` (the columns of ``beta`` are the cointegrating
vectors, i.e. basket weights).

Estimation (reduced-rank regression)
------------------------------------
Regress ``dy_t`` and ``y_{t-1}`` on the lagged differences (and deterministic terms); call the
residuals ``R0`` and ``R1`` and set ``S_ij = R_i' R_j / T``.  The squared canonical correlations
``lambda_1 >= ... >= lambda_n`` are the eigenvalues of ``S11^-1 S10 S00^-1 S01``; the
eigenvectors, normalised so ``V' S11 V = I``, span the cointegrating space.

The two sequential tests of H0: ``rank <= r`` against H1:

* **Trace**:              ``LR_trace(r) = -T * sum_{i=r+1..n} ln(1 - lambda_i)``   (H1: rank > r)
* **Maximum eigenvalue**: ``LR_max(r)   = -T * ln(1 - lambda_{r+1})``              (H1: rank = r+1)

Their null distributions are non-standard (functionals of Brownian motion); critical values are the
MacKinnon-Haug-Michelis (1999) tables shipped with statsmodels, used here as data.  The rank is the
first ``r`` (from 0 upward) at which the null is *not* rejected.

Deterministic terms (``det_order``): ``-1`` none, ``0`` constant, ``1`` linear trend.  As in the
statsmodels reference, the polynomial is projected out of the levels first and the constant out of
the differences.

**``det_order`` is an assumption about the data; the wrong one breaks the size.**  ``0`` is the
unrestricted-constant model: the critical values assume the series *drift* (their last row is
exactly chi-square(1), which is what drift produces).  Applied to driftless series it over-rejects
the last rank test badly (69 % correct rank vs 95 % with ``-1`` in the tests' driftless rank-2
system).  Log prices drift, so ``0`` is the natural choice for baskets of them; ``-1`` is for
demeaned, driftless data.  Report the rank decision together with the ``det_order`` used.

Caveats
-------
* Validation: for ``k_ar_diff >= 1`` the statistics, eigenvalues, eigenvectors and critical values
  agree with ``statsmodels.coint_johansen`` to ~1e-9.  For ``k_ar_diff = 0`` statsmodels pairs
  ``dy_t`` with ``y_t``, whereas Johansen's VAR(1) form (used here, and checked against an
  independent brute-force computation in the tests) uses ``y_{t-1}``; the statistics then differ
  (e.g. 321.2 vs 325.5 for the first trace statistic in the test system).  Both over-reject
  similarly under a true null at T=250 (8.1 % vs 7.3 % at a nominal 5 %, N=3, 1000 draws), so no
  claim is made about which is better in size.
* Small-sample size distortion is large: with many series and few observations the trace test
  over-rejects, so ``n`` should be small relative to ``T`` (see the size study in the tests).
* The result depends on ``k`` (lagged differences); the test does not choose it.
* Cointegrating vectors are identified only up to scale and rotation when ``r > 1``; ``V`` is
  normalised, not "the" economic basket.  Use ``normalised_vector`` for an interpretable scaling.
* Each rank test on a candidate basket is a hypothesis test and must be counted (Stage 8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.coint_tables import c_sja, c_sjt

_LEVELS = {"90%": 0, "95%": 1, "99%": 2}


@dataclass(frozen=True)
class JohansenResult:
    eigenvalues: np.ndarray  # lambda_1 >= ... >= lambda_n
    eigenvectors: np.ndarray  # columns: cointegrating vectors, V' S11 V = I
    trace_stat: np.ndarray  # LR_trace(r) for r = 0..n-1
    max_eig_stat: np.ndarray  # LR_max(r)   for r = 0..n-1
    trace_crit: np.ndarray  # (n, 3): 90/95/99 %
    max_eig_crit: np.ndarray
    nobs: int  # effective sample size T
    det_order: int
    k_ar_diff: int

    def rank(self, level: str = "95%", statistic: str = "trace") -> int:
        """Cointegration rank: first ``r`` for which H0 "rank <= r" is not rejected."""
        stat, crit = (
            (self.trace_stat, self.trace_crit)
            if statistic == "trace"
            else (self.max_eig_stat, self.max_eig_crit)
        )
        col = _LEVELS[level]
        for r in range(len(stat)):
            if stat[r] <= crit[r, col]:
                return r
        return len(stat)

    def normalised_vector(self, i: int = 0, on: int = 0) -> np.ndarray:
        """Cointegrating vector ``i`` scaled so component ``on`` equals 1."""
        v = self.eigenvectors[:, i]
        return v / v[on]


def _polynomial_detrend(y: np.ndarray, order: int) -> np.ndarray:
    if order == -1:
        return y
    basis = np.vander(np.linspace(-1, 1, len(y)), order + 1)
    return y - basis @ np.linalg.lstsq(basis, y, rcond=None)[0]


def _residualise(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return y
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def johansen_test(endog, det_order: int = 0, k_ar_diff: int = 1) -> JohansenResult:
    """Johansen trace / max-eigenvalue tests on the columns of ``endog`` (levels, not returns)."""
    y = np.asarray(endog, dtype=float)
    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError("endog must be a 2-d array with at least two series")
    if not np.all(np.isfinite(y)):
        raise ValueError("endog contains NaN/inf; align and drop missing rows first")
    if det_order not in (-1, 0, 1):
        raise ValueError("det_order must be -1, 0 or 1 (critical values exist only for these)")
    if k_ar_diff < 0:
        raise ValueError("k_ar_diff must be >= 0")
    n_obs, n = y.shape
    if n > 12:
        raise ValueError("critical values are only tabulated for up to 12 series")
    if n_obs - k_ar_diff - 1 <= n * (k_ar_diff + 2):
        raise ValueError("sample too short for this many series and lags")

    concentrate = 0 if det_order > -1 else -1  # constants are projected out of the differences
    y = _polynomial_detrend(y, det_order)
    dy = np.diff(y, axis=0)
    k = k_ar_diff
    if k > 0:
        z = np.hstack([dy[k - i : len(dy) - i] for i in range(1, k + 1)])
        z = _polynomial_detrend(z, concentrate)
    else:
        z = np.empty((len(dy), 0))
    dy_t = _polynomial_detrend(dy[k:], concentrate)
    y_lag = _polynomial_detrend(y[k : n_obs - 1], concentrate)  # y_{t-1}, aligned with dy[k:]

    r0 = _residualise(dy_t, z)
    r1 = _residualise(y_lag, z)
    t = r1.shape[0]
    s00, s01, s11 = r0.T @ r0 / t, r0.T @ r1 / t, r1.T @ r1 / t

    # symmetric form of the generalised eigenproblem: stable, real, ordered eigenvalues
    w, u = np.linalg.eigh(s11)
    s11_inv_half = u @ np.diag(w**-0.5) @ u.T
    m = s11_inv_half @ s01.T @ np.linalg.solve(s00, s01) @ s11_inv_half
    lam, vec = np.linalg.eigh((m + m.T) / 2)
    order = np.argsort(lam)[::-1]
    lam = np.clip(lam[order], 0.0, 1.0 - 1e-15)
    v = s11_inv_half @ vec[:, order]  # V' S11 V = I
    v = v * np.where(v[np.argmax(np.abs(v), axis=0), np.arange(n)] < 0, -1.0, 1.0)  # stable sign

    log_terms = np.log1p(-lam)
    trace = -t * np.cumsum(log_terms[::-1])[::-1]
    max_eig = -t * log_terms
    cv_trace = np.array([c_sjt(n - i, det_order) for i in range(n)])
    cv_max = np.array([c_sja(n - i, det_order) for i in range(n)])
    return JohansenResult(lam, v, trace, max_eig, cv_trace, cv_max, t, det_order, k_ar_diff)
