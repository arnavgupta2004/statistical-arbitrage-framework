"""Sup-F mean-break test (size, power, location, agreement with a brute-force loop) and the
subspace-overlap measure."""

from __future__ import annotations

import numpy as np
import pytest

from statarb.statistics.breaks import _sup_f, subspace_overlap, sup_mean_break_test


def test_sup_f_matches_a_brute_force_loop():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 80))
    x[1, 40:] += 1.5
    lo, hi = 12, 68
    stat, loc = _sup_f(x, lo, hi)
    for r in range(3):
        best, at = -1.0, None
        for k in range(lo, hi + 1):
            a, b = x[r, :k], x[r, k:]
            ssr1 = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
            ssr0 = ((x[r] - x[r].mean()) ** 2).sum()
            f = (ssr0 - ssr1) / (ssr1 / (80 - 2))
            if f > best:
                best, at = f, k
        assert stat[r] == pytest.approx(best, rel=1e-10) and loc[r] == at


def test_a_clear_break_is_found_and_located():
    rng = np.random.default_rng(1)
    hits, err = 0, []
    for i in range(60):
        x = rng.normal(0, 1, 400)
        x[250:] += 1.0
        out = sup_mean_break_test(x, n_boot=300, seed=i, mean_block=1.0)
        hits += out["p_value"] < 0.05
        err.append(abs(out["index"] - 250) / 400)
        assert out["mean_after"] > out["mean_before"]
    assert hits / 60 > 0.9 and np.median(err) < 0.05


@pytest.mark.parametrize("phi,block", [(0.0, 1.0), (0.3, 10.0)])
def test_size_under_no_break_iid_and_serially_dependent(phi, block):
    rng = np.random.default_rng(2)
    rej, sims, n = 0, 250, 300
    for i in range(sims):
        e = rng.normal(size=n)
        x = np.empty(n)
        x[0] = e[0]
        for t in range(1, n):
            x[t] = phi * x[t - 1] + e[t]
        rej += sup_mean_break_test(x, n_boot=300, seed=i, mean_block=block)["p_value"] < 0.05
    assert 0.01 <= rej / sims <= 0.11


def test_trimming_and_validation():
    x = np.random.default_rng(3).normal(size=100)
    x[3:] += 5.0  # a break outside the trimmed interior cannot be located there
    out = sup_mean_break_test(x, trim=0.15, n_boot=200, seed=1)
    assert 15 <= out["index"] <= 85
    for kw in ({"trim": 0.0}, {"trim": 0.5}):
        with pytest.raises(ValueError):
            sup_mean_break_test(x, **kw)
    with pytest.raises(ValueError):
        sup_mean_break_test(x[:10])
    bad = x.copy()
    bad[5] = np.nan
    with pytest.raises(ValueError):
        sup_mean_break_test(bad)
    assert sup_mean_break_test(x, seed=4, n_boot=100) == sup_mean_break_test(x, seed=4, n_boot=100)


def test_subspace_overlap_identities():
    e = np.eye(6)
    assert subspace_overlap(e[:, :2], e[:, :2]) == pytest.approx(1.0)
    assert subspace_overlap(e[:, :2], e[:, 2:4]) == pytest.approx(0.0, abs=1e-15)
    assert subspace_overlap(e[:, :2], e[:, [1, 2]]) == pytest.approx(
        0.5
    )  # one shared direction of two
    rng = np.random.default_rng(4)
    v = rng.normal(size=(30, 3))
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    assert subspace_overlap(v, v @ rot * 2.5) == pytest.approx(1.0)  # same span, any basis
    assert subspace_overlap(v, e[:30, :3] if False else rng.normal(size=(30, 3))) == pytest.approx(
        subspace_overlap(rng.normal(size=(30, 3)), v), abs=0.35
    )
    rand = np.mean(
        [subspace_overlap(rng.normal(size=(200, 5)), rng.normal(size=(200, 5))) for _ in range(60)]
    )
    assert rand == pytest.approx(5 / 200, rel=0.25)  # two random subspaces overlap by about k / N
    a, b = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
    assert subspace_overlap(a, b) == pytest.approx(subspace_overlap(b, a))
    with pytest.raises(ValueError):
        subspace_overlap(rng.normal(size=(20, 3)), rng.normal(size=(20, 2)))


def test_step_series_p_value_floor_and_ties():
    step = np.r_[np.zeros(100), np.ones(100)] + np.random.default_rng(0).normal(0, 0.01, 200)
    out = sup_mean_break_test(step, n_boot=400, seed=1)
    assert out["index"] == 100  # x[:100] and x[100:] are the two regimes
    assert out["mean_before"] == pytest.approx(step[:100].mean()) and out[
        "mean_after"
    ] == pytest.approx(step[100:].mean())
    assert out["p_value"] == pytest.approx(1 / 401)  # a p-value is never zero
    flat = sup_mean_break_test(
        np.full(100, 3.0), n_boot=200, seed=1
    )  # nothing to find: every draw ties
    assert flat["statistic"] == 0.0 and flat["p_value"] == 1.0
