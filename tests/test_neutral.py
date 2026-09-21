"""Sizing and neutralisation of the residual book."""

from __future__ import annotations

import numpy as np
import pytest

from statarb.portfolio.neutral import BookConfig, neutral_book, neutralise, size_weights


def test_projection_is_exactly_neutral_minimal_and_idempotent():
    rng = np.random.default_rng(0)
    n, m = 40, 5
    b = rng.normal(0, 1, (n, m))
    w = rng.normal(0, 0.02, n)
    out = neutralise(w, b)
    assert out.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(b.T @ out, 0.0, atol=1e-12)
    g = np.column_stack([np.ones(n), b])
    assert np.allclose(g.T @ out, 0.0, atol=1e-12)
    # what was removed lies in the span of the constraints => the smallest possible change
    assert np.allclose(w - out, g @ np.linalg.lstsq(g, w - out, rcond=None)[0], atol=1e-12)
    # any other feasible book is farther from w
    for _ in range(20):
        other = neutralise(w + rng.normal(0, 0.05, n), b)
        assert np.linalg.norm(other - w) >= np.linalg.norm(out - w) - 1e-12
    assert np.allclose(neutralise(out, b), out, atol=1e-12)


def test_no_factor_case_is_dollar_neutral_only():
    w = np.array([0.02, 0.01, -0.005, 0.0])
    out = neutralise(w, np.zeros((4, 0)))
    assert out.sum() == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(out, w - w.mean())  # the minimal change that removes the net dollar exposure


def test_no_factor_book_is_dollar_neutral():
    d, n = 4, 20
    state = np.zeros((d, n), dtype=np.int8)
    state[1, :6] = 1  # net long unless neutralised
    state[2, :6], state[2, 6:8] = 1, -1
    beta = np.zeros((d, n, 0))
    w = neutral_book(state, np.full((d, n), 0.01), beta, np.ones((d, n), dtype=bool), BookConfig())
    assert np.abs(w.sum(axis=1)).max() < 1e-12 and np.abs(w[1]).sum() > 0


def test_sizing_is_inverse_residual_vol_clipped_and_signed():
    state = np.array([[1, -1, 1, 0, 1]], dtype=np.int8)
    sig = np.array([[0.01, 0.02, 0.005, 0.01, 0.0001]])
    cfg = BookConfig(name_size=0.02, max_mult=2.5)
    w = size_weights(state, sig, cfg)[0]
    ref = np.median(sig)  # 0.01
    assert w[0] == pytest.approx(0.02 * ref / 0.01)
    assert w[1] == pytest.approx(-0.02 * ref / 0.02)
    assert w[2] == pytest.approx(0.02 * 2.0)
    assert w[3] == 0.0
    assert w[4] == pytest.approx(0.02 * 2.5)  # clipped
    nan_sig = np.array([[0.01, np.nan, 0.01, 0.01, 0.01]])
    assert size_weights(state, nan_sig, cfg)[0, 1] == 0.0  # no residual vol -> no position


def test_neutral_book_flat_days_stay_flat_and_untradable_names_get_nothing():
    rng = np.random.default_rng(1)
    d, n, k = 6, 30, 3
    beta = rng.normal(1, 0.5, (d, n, k))
    sig = np.full((d, n), 0.01)
    state = np.zeros((d, n), dtype=np.int8)
    state[2, :8] = 1
    state[2, 8:16] = -1
    state[3, :8] = 1
    tradable = np.ones((d, n), dtype=bool)
    tradable[2, 3] = False
    w = neutral_book(state, sig, beta, tradable, BookConfig())
    assert np.all(w[[0, 1, 4, 5]] == 0.0)  # no positions -> no hedge put on for nothing
    assert w[2, 3] == 0.0 and w[2].sum() == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(beta[2].T @ w[2], 0.0, atol=1e-10)
    assert np.allclose(beta[3].T @ w[3], 0.0, atol=1e-10) and np.abs(w[3]).sum() > 0
    # the signal survives neutralisation: longs are still net long, shorts net short
    assert w[2, :8].sum() > 0 > w[2, 8:16].sum()


def test_names_without_betas_are_excluded_from_the_hedge():
    d, n, k = 1, 12, 2
    rng = np.random.default_rng(2)
    beta = rng.normal(1, 0.5, (d, n, k))
    beta[0, 5] = np.nan
    sig = np.full((d, n), 0.01)
    state = np.zeros((d, n), dtype=np.int8)
    state[0, :3], state[0, 6:9] = 1, -1
    w = neutral_book(state, sig, beta, np.ones((d, n), dtype=bool), BookConfig())
    assert w[0, 5] == 0.0
    ok = np.isfinite(beta[0]).all(axis=1)
    assert np.allclose(beta[0][ok].T @ w[0][ok], 0.0, atol=1e-10)
