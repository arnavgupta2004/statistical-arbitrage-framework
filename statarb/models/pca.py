"""Principal-component factor model of a return panel.

Given a training window of daily returns ``R`` (T x N), standardise each stock (subtract its mean,
divide by its standard deviation) to ``Z`` and eigen-decompose the sample correlation matrix::

    C = Z'Z / (T - 1) = V diag(lambda) V'          lambda_1 >= ... >= lambda_N

The first ``k`` eigenvectors span the *factor space*.  Factor (eigenportfolio) returns and the
decomposition ``R = B F + eps`` are::

    F = Z V_k                 (T x k)      each column has variance lambda_j
    B = V_k                   (N x k)      loadings on standardised returns
    eps = Z - F V_k'          (T x N)      residual, exactly orthogonal to F (in sample)

so ``sum(lambda_1..k) / N`` is the share of variance explained.  The fit uses *only the rows it is
given*: causality is the caller's job (``signals/residuals.py`` passes trailing windows that end
before the trading day) and is checked by the tests.

Choosing ``k``: a fixed number, or the number of eigenvalues above the Marchenko-Pastur upper edge
``(1 + sqrt(N/T))^2`` -- what pure noise of the same shape would produce.  Eigenvector signs are
arbitrary; they are normalised so the largest-magnitude component of each is positive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PCAFit:
    tickers: list[str]
    mean: np.ndarray  # (N,)
    scale: np.ndarray  # (N,)  standard deviations used to standardise
    eigenvalues: np.ndarray  # (N,) descending, of the correlation matrix
    components: np.ndarray  # (N, N) eigenvectors as columns
    n_obs: int

    def explained_ratio(self, k: int) -> float:
        return float(self.eigenvalues[:k].sum() / self.eigenvalues.sum())

    def standardise(self, returns) -> np.ndarray:
        """Standardise with the *fitted* mean and scale (never re-estimated on new data)."""
        return (np.asarray(returns, dtype=float) - self.mean) / self.scale

    def factor_returns(self, returns, k: int) -> np.ndarray:
        """Eigenportfolio returns ``Z V_k`` for any rows of returns (T x N) -> (T x k)."""
        return self.standardise(returns) @ self.components[:, :k]

    def loadings(self, k: int) -> np.ndarray:
        return self.components[:, :k]

    def residuals(self, returns, k: int) -> np.ndarray:
        """Standardised residual ``Z - Z V_k V_k'``: what the first ``k`` factors do not explain."""
        z = self.standardise(returns)
        v = self.components[:, :k]
        return z - (z @ v) @ v.T

    def marchenko_pastur_k(self) -> int:
        return n_factors_marchenko_pastur(self.eigenvalues, self.n_obs)


def marchenko_pastur_upper(n_obs: int, n_assets: int) -> float:
    """Upper edge of the eigenvalue spectrum of a correlation matrix of iid noise."""
    return float((1.0 + np.sqrt(n_assets / n_obs)) ** 2)


def n_factors_marchenko_pastur(eigenvalues: np.ndarray, n_obs: int) -> int:
    return int((eigenvalues > marchenko_pastur_upper(n_obs, len(eigenvalues))).sum())


def fit_pca(returns: pd.DataFrame | np.ndarray, min_obs_ratio: float = 1.0) -> PCAFit:
    """Fit on a gap-free (T x N) return matrix; requires ``T >= min_obs_ratio * N``."""
    tickers = list(returns.columns) if isinstance(returns, pd.DataFrame) else None
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("returns must be a finite 2-d array (no NaN; drop or fix gaps first)")
    t, n = x.shape
    if t < max(30, min_obs_ratio * n):
        raise ValueError(f"too few observations ({t}) for {n} assets")
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=1)
    if np.any(scale <= 0):
        raise ValueError("a return series has zero variance")
    z = (x - mean) / scale
    corr = z.T @ z / (t - 1)
    w, v = np.linalg.eigh((corr + corr.T) / 2.0)
    order = np.argsort(w)[::-1]
    w, v = np.clip(w[order], 0.0, None), v[:, order]
    flip = np.sign(v[np.argmax(np.abs(v), axis=0), np.arange(n)])  # largest component positive
    v = v * np.where(flip == 0, 1.0, flip)
    return PCAFit(
        tickers if tickers is not None else [str(i) for i in range(n)], mean, scale, w, v, t
    )
