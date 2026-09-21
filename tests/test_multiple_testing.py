"""Multiple-testing procedures: exact agreement with statsmodels (FDR), the law of the vectorised
bootstrap, the size and power of the reality check / SPA / Romano-Wolf by Monte Carlo, and the
CSCV probability of backtest overfitting against a brute-force enumeration."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from statarb.statistics.bootstrap import stationary_bootstrap_indices, stationary_bootstrap_matrix
from statarb.statistics.multiple_testing import (
    benjamini_hochberg,
    benjamini_yekutieli,
    cscv_pbo,
    reality_check,
    romano_wolf,
    spa_test,
)


# ---- FDR --------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_bh_and_by_match_statsmodels_exactly(seed):
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(0, 0.01, 8), rng.uniform(0, 1, 40), [0.5, 0.5, 1.0]])
    rng.shuffle(p)
    for fn, name in ((benjamini_hochberg, "fdr_bh"), (benjamini_yekutieli, "fdr_by")):
        for q in (0.05, 0.2):
            rej, adj = fn(p, q)
            ref_rej, ref_adj, *_ = multipletests(p, alpha=q, method=name)
            assert np.array_equal(rej, ref_rej) and np.allclose(adj, ref_adj, atol=1e-12)


def test_bh_controls_the_false_discovery_rate():
    rng = np.random.default_rng(0)
    m, m1, reps, q = 100, 20, 1500, 0.10
    fdp, any_null = [], 0
    for _ in range(reps):
        p = np.concatenate([rng.uniform(0, 1, m - m1), rng.beta(0.05, 30, m1)])
        rej, _ = benjamini_hochberg(p, q)
        fdp.append(rej[: m - m1].sum() / max(rej.sum(), 1))
        any_null += benjamini_hochberg(rng.uniform(0, 1, m), q)[0].any()
    assert np.mean(fdp) <= q * (m - m1) / m * 1.25 + 0.01  # BH gives FDR <= q * m0 / m
    assert any_null / reps == pytest.approx(q, abs=0.03)  # under the global null: FWER ~ q


def test_bh_edge_cases():
    rej, adj = benjamini_hochberg(np.array([0.9]), 0.05)
    assert not rej[0] and adj[0] == 0.9
    rej, _ = benjamini_hochberg(np.array([0.001, 0.002, 0.003]), 0.05)
    assert rej.all()
    assert (
        benjamini_yekutieli(np.array([0.01] * 10), 0.05)[1].max()
        >= benjamini_hochberg(np.array([0.01] * 10), 0.05)[1].max()
    )


# ---- the vectorised stationary bootstrap ------------------------------------------------------
def test_vectorised_bootstrap_has_the_stationary_bootstrap_law():
    rng = np.random.default_rng(1)
    n, b, mean_block = 300, 400, 10.0
    idx = stationary_bootstrap_matrix(n, b, mean_block, rng)
    assert idx.shape == (b, n) and idx.min() >= 0 and idx.max() < n
    cont = (
        idx[:, 1:] == (idx[:, :-1] + 1) % n
    )  # a block continues, or a restart lands on it by chance
    assert cont.mean() == pytest.approx(1 - 1 / mean_block + 1 / (mean_block * n), abs=0.006)
    loop = np.array([stationary_bootstrap_indices(n, mean_block, rng) for _ in range(b)])
    assert (loop[:, 1:] == (loop[:, :-1] + 1) % n).mean() == pytest.approx(cont.mean(), abs=0.01)
    first_last = stationary_bootstrap_matrix(60, 6000, mean_block, rng)[:, [0, -1]]
    for col in (0, 1):  # every date equally likely at any position (independent across rows)
        assert stats.chisquare(np.bincount(first_last[:, col], minlength=60)).pvalue > 0.001
    assert np.array_equal(
        stationary_bootstrap_matrix(50, 3, 5, np.random.default_rng(9)),
        stationary_bootstrap_matrix(50, 3, 5, np.random.default_rng(9)),
    )
    same_block = stationary_bootstrap_matrix(20, 5, 1e9, np.random.default_rng(0))
    assert np.array_equal(
        np.diff(same_block, axis=1) % 20, np.ones((5, 19), dtype=int)
    )  # one long block
    with pytest.raises(ValueError):
        stationary_bootstrap_matrix(1, 5, 5, np.random.default_rng(0))


# ---- family-wise tests: size and power --------------------------------------------------------
def family(rng, n, k, sr_daily=None, n_poor=0, poor_sr=-0.15):
    """k strategies sharing a common factor; strategy 0 gets Sharpe ``sr_daily``; ``n_poor`` are bad."""
    common = rng.normal(size=(n, 1))
    x = 0.6 * common + 0.8 * rng.normal(size=(n, k))
    if sr_daily is not None:
        x[:, 0] += sr_daily
    if n_poor:
        x[:, k - n_poor :] += poor_sr
    return x


def test_reality_check_and_spa_have_the_right_size_and_power():
    rng = np.random.default_rng(5)
    sims, n, k = 250, 400, 10
    rc = spa = 0
    for i in range(sims):
        x = family(rng, n, k)
        rc += reality_check(x, n_boot=300, seed=i)["p_value"] < 0.05
        spa += spa_test(x, n_boot=300, seed=i)["consistent"] < 0.05
    assert 0.01 <= rc / sims <= 0.10 and 0.01 <= spa / sims <= 0.11
    rc = spa = 0
    for i in range(80):
        x = family(rng, n, k, sr_daily=0.24)  # z ~ 4.8 on one of ten
        rc += reality_check(x, n_boot=300, seed=i)["p_value"] < 0.05
        spa += spa_test(x, n_boot=300, seed=i)["consistent"] < 0.05
    assert rc / 80 > 0.9 and spa / 80 > 0.9


def test_spa_is_not_diluted_by_poor_strategies_but_the_reality_check_is():
    rng = np.random.default_rng(6)
    n, k, poor, sims = 400, 40, 34, 120
    rc = spa = 0
    for i in range(sims):
        x = family(rng, n, k, sr_daily=0.11, n_poor=poor, poor_sr=-0.15)
        rc += reality_check(x, n_boot=300, seed=i)["p_value"] < 0.05
        spa += spa_test(x, n_boot=300, seed=i)["consistent"] < 0.05
    assert spa / sims > rc / sims + 0.05


def test_spa_p_values_are_ordered_and_report_poor_strategies():
    rng = np.random.default_rng(7)
    for i in range(10):
        x = family(rng, 300, 12, sr_daily=0.05 * (i % 3), n_poor=3)
        out = spa_test(x, n_boot=400, seed=i)
        assert out["lower"] <= out["consistent"] + 1e-12 <= out["upper"] + 2e-12
        assert out["n_poor"] >= 0


def test_one_strategy_reality_check_is_the_one_sided_t_test():
    rng = np.random.default_rng(8)
    x = rng.normal(0.05, 1.0, (600, 1))
    p = reality_check(x, n_boot=4000, mean_block=1.0, seed=1)["p_value"]
    t = stats.ttest_1samp(x[:, 0], 0.0, alternative="greater").pvalue
    assert p == pytest.approx(t, abs=0.04)
    assert reality_check(x, n_boot=500, seed=3)["best"] == 0


def test_reality_check_is_deterministic_and_rejects_bad_input():
    x = np.random.default_rng(0).normal(size=(200, 4))
    assert reality_check(x, n_boot=200, seed=4) == reality_check(x, n_boot=200, seed=4)
    x[3, 1] = np.nan
    with pytest.raises(ValueError):
        reality_check(x)


def test_romano_wolf_adjusts_upward_controls_fwer_and_finds_the_real_one():
    rng = np.random.default_rng(9)
    n, k, sims = 400, 12, 200
    any_rej = 0
    for i in range(sims):
        out = romano_wolf(family(rng, n, k), n_boot=300, seed=i)
        assert (out["p_adjusted"] >= out["p_raw"] - 1e-12).all()
        any_rej += (out["p_adjusted"] < 0.05).any()
    assert any_rej / sims <= 0.10  # family-wise error ~ 5 %
    found = only_it = 0
    for i in range(60):
        out = romano_wolf(family(rng, n, k, sr_daily=0.24), n_boot=300, seed=i)
        found += out["p_adjusted"][0] < 0.05
        only_it += (out["p_adjusted"][1:] < 0.05).sum() == 0
    assert found / 60 > 0.9 and only_it / 60 > 0.9
    one = romano_wolf(rng.normal(0.02, 1, (300, 1)), n_boot=300, seed=1)
    assert one["p_adjusted"][0] == pytest.approx(one["p_raw"][0])


def test_romano_wolf_adjusted_p_values_are_monotone_in_the_statistic():
    x = family(np.random.default_rng(10), 300, 8, sr_daily=0.08)
    out = romano_wolf(x, n_boot=400, seed=2)
    ordered = out["p_adjusted"][out["order"]]
    assert (np.diff(ordered) >= -1e-12).all()


# ---- CSCV / PBO -------------------------------------------------------------------------------
def brute_force_pbo(x, s):
    t, n = x.shape
    edges = np.linspace(0, t, s + 1).astype(int)
    blocks = [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    below = total = 0
    for ins in combinations(range(s), s // 2):
        rows_in = np.concatenate([blocks[i] for i in ins])
        rows_out = np.concatenate([blocks[i] for i in range(s) if i not in ins])
        sr_in = x[rows_in].mean(0) / x[rows_in].std(0, ddof=1)
        sr_out = x[rows_out].mean(0) / x[rows_out].std(0, ddof=1)
        best = int(np.argmax(sr_in))
        rank = 1 + np.sum(sr_out < sr_out[best]) + 0.5 * (np.sum(sr_out == sr_out[best]) - 1)
        omega = rank / (n + 1)
        below += np.log(omega / (1 - omega)) <= 0
        total += 1
    return below / total, total


@pytest.mark.parametrize("s,n", [(4, 3), (6, 5), (8, 4)])
def test_cscv_matches_a_brute_force_enumeration(s, n):
    x = np.random.default_rng(s * 10 + n).normal(0.0005, 0.01, (240, n))
    out = cscv_pbo(x, s)
    ref, total = brute_force_pbo(x, s)
    assert out["n_splits"] == total and out["pbo"] == pytest.approx(ref, abs=1e-12)


def test_pbo_is_half_for_noise_and_near_zero_for_a_persistent_winner():
    rng = np.random.default_rng(11)
    noise = np.mean([cscv_pbo(rng.normal(0, 0.01, (800, 10)), 16)["pbo"] for _ in range(6)])
    assert 0.35 < noise < 0.65
    x = rng.normal(0, 0.01, (800, 10))
    x[:, 3] += 0.0025  # a genuinely better strategy, ~4 annualised
    out = cscv_pbo(x, 16)
    assert out["pbo"] < 0.1 and out["is_oos_slope"] == out["is_oos_slope"]
    # picking the in-sample winner from noise: it is no better than average out of sample
    noise_out = cscv_pbo(rng.normal(0, 0.01, (800, 30)), 16)
    assert noise_out["mean_is_best_sharpe"] > 0.05  # per-day units: the winner looks good in-sample
    assert noise_out["mean_oos_sharpe_of_is_best"] < 0.02  # ... and is not, out of sample


def test_pbo_input_validation():
    x = np.random.default_rng(0).normal(size=(200, 5))
    for bad in (3, 5, 2):
        with pytest.raises(ValueError):
            cscv_pbo(x, bad)
    with pytest.raises(ValueError):
        cscv_pbo(x[:, :1], 4)


def exact(rng, n, means, sd=1.0):
    """Columns with exactly the given means and standard deviation (a controlled t-statistic)."""
    z = rng.normal(size=(n, len(means)))
    z = (z - z.mean(0)) / z.std(0, ddof=1)
    return z * np.asarray(sd) + np.asarray(means)


def test_bootstrap_p_values_never_reach_zero_and_the_t_scale_is_the_usual_one():
    rng = np.random.default_rng(13)
    x = exact(rng, 300, [10.0])
    assert reality_check(x, n_boot=500, seed=1)["p_value"] == pytest.approx(1 / 501)
    assert romano_wolf(x, n_boot=500, seed=1)["p_raw"][0] == pytest.approx(1 / 501)
    y = rng.normal(0.08, 1.0, (600, 1))
    t_stat = stats.ttest_1samp(y[:, 0], 0.0).statistic
    assert romano_wolf(y, n_boot=1000, mean_block=1.0, seed=2)["t"][0] == pytest.approx(
        t_stat, rel=0.05
    )


def test_spa_recentres_only_clearly_poor_strategies():
    x = exact(np.random.default_rng(14), 400, [-0.05, -0.20, 0.0, 0.0])  # t = -1, -4, 0, 0
    out = spa_test(x, n_boot=500, mean_block=1.0, seed=1)
    assert out["n_poor"] == 1  # the threshold is -sqrt(2 log log n) = -1.9, not zero


def test_spa_is_one_when_no_strategy_looks_positive():
    x = exact(np.random.default_rng(15), 300, [-0.05, -0.1, -0.02])
    out = spa_test(x, n_boot=400, seed=1)
    assert out["statistic"] == 0.0 and out["lower"] == out["consistent"] == out["upper"] == 1.0


def test_spa_is_scale_invariant_and_the_reality_check_is_not():
    rng = np.random.default_rng(16)
    x = family(rng, 400, 6, sr_daily=0.2)
    scaled = x * np.array(
        [0.1, 1.0, 1.0, 3.0, 1.0, 1.0]
    )  # the one with the edge becomes tiny in RC's eyes
    assert spa_test(scaled, n_boot=400, seed=3)["consistent"] == pytest.approx(
        spa_test(x, n_boot=400, seed=3)["consistent"], abs=1e-9
    )
    assert (
        reality_check(scaled, n_boot=400, seed=3)["p_value"]
        > reality_check(x, n_boot=400, seed=3)["p_value"] + 0.2
    )
    assert romano_wolf(scaled, n_boot=400, seed=3)["p_adjusted"][0] == pytest.approx(
        romano_wolf(x, n_boot=400, seed=3)["p_adjusted"][0], abs=1e-9
    )


def test_romano_wolf_enforces_monotone_adjusted_p_values():
    x = exact(np.random.default_rng(17), 400, [2.0 / 20, 1.9 / 20])  # t ~ 2.0 and 1.9, independent
    out = romano_wolf(x, n_boot=4000, mean_block=1.0, seed=1)
    top, second = out["order"]
    assert out["p_raw"][second] < out["p_adjusted"][top]  # the weaker one alone would look better
    assert out["p_adjusted"][second] == out["p_adjusted"][top]  # ... but is never adjusted below it


def test_romano_wolf_stepdown_shrinks_the_comparison_set():
    """A huge effect must stop counting against the weaker one: single-step would give ~2x the p."""
    x = exact(np.random.default_rng(18), 400, [6.0 / 20, 2.0 / 20])  # t ~ 6 and 2
    out = romano_wolf(x, n_boot=6000, mean_block=1.0, seed=1)
    weak = int(np.argmin(out["t"]))
    assert out["p_raw"][weak] == pytest.approx(0.0228, abs=0.01)
    assert out["p_adjusted"][weak] == pytest.approx(out["p_raw"][weak], abs=1e-12)  # not 0.045
