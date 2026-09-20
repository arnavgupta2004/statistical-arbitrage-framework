"""Johansen: agreement with statsmodels (k>=1), an independent brute force (any k), simulated ranks."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_coint_rank

from statarb.selection.johansen import johansen_test


def system(n, seed, rank):
    """3 series with a known cointegration rank (0, 1 or 2)."""
    rng = np.random.default_rng(seed)
    trends = np.cumsum(rng.normal(size=(n, 3 - rank)), axis=0)
    if rank == 0:
        return trends
    if rank == 1:  # y1, y2 share trend 0 (y2 = 2 y1 + stationary); y3 has its own trend
        e = rng.normal(0, 0.5, size=(n, 2))
        return np.column_stack([trends[:, 0] + e[:, 0], 2 * trends[:, 0] + e[:, 1], trends[:, 1]])
    e = rng.normal(0, 0.5, size=(n, 3))  # rank 2: one common trend drives all three series
    return np.column_stack(
        [trends[:, 0] + e[:, 0], 2 * trends[:, 0] + e[:, 1], -trends[:, 0] + e[:, 2]]
    )


def sm_johansen(y, det, k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return coint_johansen(y, det, k)


@pytest.mark.parametrize("det", [-1, 0, 1])
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_matches_statsmodels(det, k, rank):
    y = system(400, 10 + rank, rank)
    ours, ref = johansen_test(y, det, k), sm_johansen(y, det, k)
    assert ours.eigenvalues == pytest.approx(ref.eig.real, abs=1e-9)
    assert ours.trace_stat == pytest.approx(ref.lr1, abs=1e-7)
    assert ours.max_eig_stat == pytest.approx(ref.lr2, abs=1e-7)
    assert ours.trace_crit == pytest.approx(ref.cvt) and ours.max_eig_crit == pytest.approx(ref.cvm)
    sign = np.sign((ours.eigenvectors * ref.evec.real).sum(axis=0))
    assert ours.eigenvectors * sign == pytest.approx(ref.evec.real, abs=1e-8)


@pytest.mark.parametrize("k", [1, 2])
def test_rank_decision_matches_statsmodels_vecm(k):
    y = system(500, 3, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = select_coint_rank(y, det_order=0, k_ar_diff=k, method="trace", signif=0.05)
    assert johansen_test(y, 0, k).rank("95%", "trace") == ref.rank


def brute_force_trace(y, det, k):
    """Independent textbook computation (regress out lags/constant, canonical correlations)."""
    if det == 0:
        y = y - y.mean(axis=0)
    dy = np.diff(y, axis=0)
    z = (
        np.hstack([dy[k - i : len(dy) - i] for i in range(1, k + 1)])
        if k
        else np.empty((len(dy) - k, 0))
    )
    if det == 0:
        z = z - z.mean(axis=0) if z.size else z
    dyt = dy[k:] - (dy[k:].mean(axis=0) if det == 0 else 0)
    ylag = y[k : len(y) - 1]
    ylag = ylag - (ylag.mean(axis=0) if det == 0 else 0)

    def res(a):
        return a - z @ np.linalg.lstsq(z, a, rcond=None)[0] if z.size else a

    r0, r1 = res(dyt), res(ylag)
    t = len(r0)
    s00, s01, s11 = r0.T @ r0 / t, r0.T @ r1 / t, r1.T @ r1 / t
    lam = np.sort(np.linalg.eigvals(np.linalg.solve(s11, s01.T @ np.linalg.solve(s00, s01))).real)[
        ::-1
    ]
    return -t * np.cumsum(np.log(1 - lam)[::-1])[::-1], lam


@pytest.mark.parametrize("det", [-1, 0])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_matches_an_independent_brute_force_for_every_k_including_zero(det, k):
    y = system(400, 21, 1)
    trace, lam = brute_force_trace(y, det, k)
    ours = johansen_test(y, det, k)
    assert ours.trace_stat == pytest.approx(trace, abs=1e-7)
    assert ours.eigenvalues == pytest.approx(lam, abs=1e-9)


def vecm(n, seed, rank, drift=(0.15, 0.3, -0.1), burn=200):
    """Simulate ``y_t = c + (I + alpha beta') y_{t-1} + e_t`` -- exactly the model the test assumes.

    Also returns beta.  The transition matrix is checked: ``3 - rank`` unit roots, the rest stable.
    """
    alpha, beta = {
        0: (np.zeros((3, 1)), np.zeros((3, 1))),
        1: (np.array([[-0.2], [0.1], [0.0]]), np.array([[1.0], [-1.0], [0.0]])),
        2: (
            np.array([[-0.2, 0.0], [0.0, -0.2], [0.1, 0.1]]),
            np.array([[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]]),
        ),
    }[rank]
    a = np.eye(3) + alpha @ beta.T
    moduli = np.sort(np.abs(np.linalg.eigvals(a)))[::-1]
    assert np.allclose(moduli[: 3 - rank], 1.0) and np.all(moduli[3 - rank :] < 0.99)
    rng = np.random.default_rng(seed)
    y, e = np.zeros((n + burn, 3)), rng.normal(size=(n + burn, 3))
    for t in range(1, n + burn):
        y[t] = np.asarray(drift) + a @ y[t - 1] + e[t]
    return y[burn:], beta


@pytest.mark.parametrize("rank, floor", [(0, 0.85), (1, 0.85), (2, 0.75)])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_recovers_the_true_rank_of_simulated_vecms(rank, floor, k):
    """Measured hit rates (100 draws, T=600): rank 0: 96-99 %, rank 1: 97-98 %, rank 2: 89 %."""
    hits = sum(johansen_test(vecm(600, 100 + s, rank)[0], 0, k).rank() == rank for s in range(40))
    assert hits / 40 >= floor


def test_recovers_the_cointegrating_vector_of_a_vecm():
    y, beta = vecm(4000, 5, 1)
    r = johansen_test(y, 0, 1)
    assert r.rank() == 1
    assert r.normalised_vector(0, on=0) == pytest.approx(beta[:, 0] / beta[0, 0], abs=0.05)


def test_det_order_is_an_assumption_about_drift_and_the_wrong_one_breaks_the_test():
    """Driftless rank-2 data: det_order=0 assumes drift and over-rejects the last rank test."""
    zero = (0.0, 0.0, 0.0)
    right = sum(johansen_test(vecm(600, 300 + s, 2, zero)[0], -1, 1).rank() == 2 for s in range(40))
    wrong = sum(johansen_test(vecm(600, 300 + s, 2, zero)[0], 0, 1).rank() == 2 for s in range(40))
    assert right / 40 >= 0.85 and wrong / 40 <= 0.85 and right > wrong  # measured 95 % vs 69 %
    last_crit = johansen_test(vecm(600, 1, 2)[0], 0, 1).trace_crit[-1]
    assert last_crit == pytest.approx([2.7055, 3.8415, 6.6349], abs=1e-3)  # chi-square(1): drift


def test_eigenvectors_are_normalised_and_the_statistics_are_consistent():
    r = johansen_test(system(500, 8, 1), 0, 1)
    assert np.all(np.diff(r.eigenvalues) <= 1e-12) and np.all(
        (0 <= r.eigenvalues) & (r.eigenvalues < 1)
    )
    assert r.trace_stat[-1] == pytest.approx(
        r.max_eig_stat[-1]
    )  # last trace = last max-eig by definition
    assert r.trace_stat[0] == pytest.approx(-r.nobs * np.log(1 - r.eigenvalues).sum())
    assert np.all(np.diff(r.trace_stat) <= 1e-9)  # trace statistics decrease with r
    assert r.rank("99%") <= r.rank("90%") and r.rank("95%", "maxeig") in (0, 1, 2, 3)


def test_size_is_inflated_in_small_samples_as_documented():
    """3 independent walks, T=250: rank-0 rejected in more than 5 % of draws (known distortion)."""
    rng = np.random.default_rng(1234)
    rej = 0
    for _ in range(400):
        r = johansen_test(np.cumsum(rng.normal(size=(250, 3)), axis=0), 0, 1)
        rej += r.trace_stat[0] > r.trace_crit[0, 1]
    assert 0.04 <= rej / 400 <= 0.16  # nominal 5 %, expected to sit above it


@pytest.mark.parametrize(
    "kwargs",
    [dict(det_order=2), dict(k_ar_diff=-1)],
)
def test_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        johansen_test(system(300, 1, 1), **kwargs)


def test_bad_inputs():
    with pytest.raises(ValueError, match="at least two"):
        johansen_test(np.ones((100, 1)))
    with pytest.raises(ValueError, match="NaN"):
        johansen_test(np.full((100, 2), np.nan))
    with pytest.raises(ValueError, match="too short"):
        johansen_test(system(15, 1, 1), 0, 3)
