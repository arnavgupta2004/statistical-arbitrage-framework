"""Sharpe-ratio inference that accounts for non-normality, sample length and selection.

Notation: ``sr`` is a *per-period* (here daily) Sharpe ratio, ``n`` the number of observations,
``skew`` the skewness and ``kurt`` the (non-excess) kurtosis of the returns (3 for a normal).

* **Probabilistic Sharpe ratio** (Bailey & Lopez de Prado 2012): the probability that the true
  Sharpe exceeds a benchmark ``sr_star`` given the estimate, its length and the distribution::

      PSR = Phi( (sr - sr_star) sqrt(n - 1) / sqrt(1 - skew sr + (kurt - 1)/4 sr^2) )

  Negative skew and fat tails widen the sampling distribution of ``sr`` and lower the PSR.
* **Expected maximum Sharpe of ``N`` luck-only strategies** (the selection benchmark)::

      SR0 = sqrt(V) [ (1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)) ],   g = Euler-Mascheroni

  with ``V`` the variance of the trials' Sharpe estimates: the largest of ``N`` noise estimates is
  what the best backtest looks like when nothing is there.
* **Deflated Sharpe ratio** (Bailey & Lopez de Prado 2014): ``DSR = PSR(sr_star = SR0)``.
* **Minimum track record length**: observations for a PSR of ``1 - alpha`` against ``sr_star``.
* **Minimum backtest length**: years before the best of ``N`` luck-only annualised Sharpes is
  expected to stay below a target, ``(Z_N / SR_target)^2`` with ``Z_N`` the bracket above.
* **Effective number of trials**: ``N`` correlated trials are worth fewer than ``N`` independent
  ones; two estimates are given (average correlation, participation ratio of the eigenvalues).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe_moments(r) -> tuple[float, float, float, int]:
    """Per-period Sharpe, skewness, (non-excess) kurtosis and length of a return series."""
    x = np.asarray(r, dtype=float)
    sd = x.std(ddof=1)
    if sd <= 0:
        raise ValueError("flat series: Sharpe is undefined")
    return (
        float(x.mean() / sd),
        float(stats.skew(x, bias=True)),
        float(stats.kurtosis(x, fisher=False, bias=True)),
        len(x),
    )


def sharpe_std_error(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of a per-period Sharpe estimate under the non-normal correction."""
    return float(np.sqrt((1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n - 1)))


def probabilistic_sharpe(
    sr: float, sr_star: float, n: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    return float(stats.norm.cdf((sr - sr_star) / sharpe_std_error(sr, n, skew, kurt)))


def expected_max_z(n_trials: float) -> float:
    """Expected maximum of ``N`` iid standard normals (the bracket in ``SR0``); 0 for ``N = 1``."""
    if n_trials <= 1:
        return 0.0
    return float(
        (1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    )


def expected_max_sharpe(n_trials: float, var_sr: float) -> float:
    return float(np.sqrt(var_sr) * expected_max_z(n_trials))


def deflated_sharpe(
    sr: float, n: int, skew: float, kurt: float, n_trials: float, var_sr: float
) -> dict:
    sr0 = expected_max_sharpe(n_trials, var_sr)
    return {
        "sr0": sr0,
        "dsr": probabilistic_sharpe(sr, sr0, n, skew, kurt),
        "psr_vs_zero": probabilistic_sharpe(sr, 0.0, n, skew, kurt),
    }


def min_track_record_length(
    sr: float, sr_star: float, skew: float = 0.0, kurt: float = 3.0, alpha: float = 0.05
) -> float:
    """Observations for ``PSR = 1 - alpha``; infinite if ``sr <= sr_star``."""
    if sr <= sr_star:
        return float("inf")
    z = stats.norm.ppf(1.0 - alpha)
    return float(1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) * (z / (sr - sr_star)) ** 2)


def min_backtest_years(n_trials: float, sr_annual_target: float) -> float:
    """Years after which the best of ``N`` luck-only Sharpes stops reaching ``sr_annual_target``."""
    return float((expected_max_z(n_trials) / sr_annual_target) ** 2)


def effective_trials(returns: np.ndarray) -> dict:
    """Effective number of independent trials among the columns of a (T, N) return matrix."""
    x = np.asarray(returns, dtype=float)
    n = x.shape[1]
    corr = np.corrcoef(x, rowvar=False)
    rho = float((corr.sum() - n) / (n * (n - 1)))
    lam = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    return {
        "n": n,
        "mean_pairwise_correlation": rho,
        "n_eff_average_correlation": float(rho + (1.0 - rho) * n),
        "n_eff_participation_ratio": float(lam.sum() ** 2 / (lam**2).sum()),
    }
