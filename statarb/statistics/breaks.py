"""Structural-break diagnostics.

``sup_mean_break_test``  a single unknown-date break in the *mean* of a return series (Quandt 1960,
Andrews 1993).  For every candidate date ``k`` in the trimmed interior the F statistic compares the
pooled-mean fit with the two-mean fit; the test statistic is the supremum over ``k``.  Its
distribution is non-standard (and depends on the trimming), so the p-value comes from a **stationary
bootstrap of the series**: resampling blocks destroys the *location* of any break while keeping
serial dependence and volatility clustering (the statistic is invariant to the series' mean, so
nothing needs re-centring).  The reported ``index`` is the first observation of the second regime
(``x[:index]`` and ``x[index:]`` are the two regimes).  Power against a break is low for short
samples and small changes; failing to reject is not evidence of stability.

``subspace_overlap``  how much of one ``k``-dimensional subspace lies inside another,
``||V1' V2||_F^2 / k`` in [0, 1] (1 = identical, 0 = orthogonal; two random ``k``-dimensional
subspaces of ``R^N`` overlap by about ``k / N``).  Used for the stability of PCA factor spaces.
"""

from __future__ import annotations

import numpy as np

from statarb.statistics.bootstrap import stationary_bootstrap_matrix


def _sup_f(x: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Best F statistic and its location for each row of ``x``, over candidate dates lo..hi."""
    n = x.shape[1]
    cs = np.cumsum(x, axis=1)
    total = cs[:, -1:]
    k = np.arange(lo, hi + 1)
    s1 = cs[:, lo - 1 : hi]
    m1, m2 = s1 / k, (total - s1) / (n - k)
    ssr1 = (x**2).sum(axis=1, keepdims=True) - k * m1**2 - (n - k) * m2**2
    ssr0 = (x**2).sum(axis=1, keepdims=True) - total**2 / n
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (ssr0 - ssr1) / (ssr1 / (n - 2))
    f = np.where(np.isfinite(f), f, 0.0)
    best = np.argmax(f, axis=1)
    return f[np.arange(len(x)), best], k[best]


def sup_mean_break_test(
    x, trim: float = 0.15, n_boot: int = 2000, mean_block: float = 10.0, seed: int = 0
) -> dict:
    """Sup-F test for one mean break; bootstrap p-value; ``index`` starts the second regime."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if not 0 < trim < 0.5 or n < 20 or not np.all(np.isfinite(x)):
        raise ValueError("need a finite series of >= 20 points and 0 < trim < 0.5")
    lo, hi = int(np.ceil(trim * n)), int(np.floor((1 - trim) * n))
    stat, idx = _sup_f(x[None, :], lo, hi)
    boots = x[stationary_bootstrap_matrix(n, n_boot, mean_block, np.random.default_rng(seed))]
    stat_b, _ = _sup_f(boots, lo, hi)
    return {
        "statistic": float(stat[0]),
        "index": int(idx[0]),
        "p_value": float((1 + np.sum(stat_b >= stat[0])) / (n_boot + 1)),
        "mean_before": float(x[: idx[0]].mean()),
        "mean_after": float(x[idx[0] :].mean()),
    }


def subspace_overlap(v1: np.ndarray, v2: np.ndarray) -> float:
    """``||Q1' Q2||_F^2 / k`` for the column spaces of ``v1`` and ``v2`` (each N x k)."""
    q1, q2 = np.linalg.qr(v1)[0], np.linalg.qr(v2)[0]
    if q1.shape[1] != q2.shape[1]:
        raise ValueError("subspaces must have the same dimension")
    return float(np.linalg.norm(q1.T @ q2, "fro") ** 2 / q1.shape[1])
