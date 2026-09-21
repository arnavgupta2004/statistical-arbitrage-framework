"""Sharpe inference: closed forms by hand, and Monte Carlo checks of what each formula claims
(the sampling error of a Sharpe estimate, and the size of a test that deflates for selection)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from statarb.statistics.sharpe import (
    deflated_sharpe,
    effective_trials,
    expected_max_sharpe,
    expected_max_z,
    min_backtest_years,
    min_track_record_length,
    probabilistic_sharpe,
    sharpe_moments,
    sharpe_std_error,
)


def test_psr_closed_form_and_limits():
    assert probabilistic_sharpe(0.1, 0.1, 500) == pytest.approx(0.5)
    # normal returns: Phi((sr - sr*) sqrt(n - 1) / sqrt(1 + sr^2 / 2))
    expect = stats.norm.cdf(0.1 * 10 / np.sqrt(1 + 0.5 * 0.01))
    assert probabilistic_sharpe(0.1, 0.0, 101) == pytest.approx(expect, rel=1e-12)
    assert sharpe_std_error(0.0, 101) == pytest.approx(0.1)
    # more data, higher Sharpe and positive skew help; fat tails and negative skew hurt
    base = probabilistic_sharpe(0.08, 0.0, 250, 0.0, 3.0)
    assert probabilistic_sharpe(0.08, 0.0, 500, 0.0, 3.0) > base
    assert probabilistic_sharpe(0.10, 0.0, 250, 0.0, 3.0) > base
    assert (
        probabilistic_sharpe(0.08, 0.0, 250, 1.0, 3.0)
        > base
        > probabilistic_sharpe(0.08, 0.0, 250, -1.0, 3.0)
    )
    assert probabilistic_sharpe(0.08, 0.0, 250, 0.0, 9.0) < base


def test_sharpe_moments_match_scipy_and_flat_series_is_rejected():
    r = np.random.default_rng(0).standard_t(5, 2000) * 0.01 + 0.0005
    sr, sk, ku, n = sharpe_moments(r)
    assert sr == pytest.approx(r.mean() / r.std(ddof=1)) and n == 2000
    assert sk == pytest.approx(stats.skew(r)) and ku == pytest.approx(stats.kurtosis(r) + 3.0)
    with pytest.raises(ValueError):
        sharpe_moments(np.zeros(10))


def _skewed(rng, size):
    """Negatively skewed, fat-tailed returns with a clearly positive Sharpe."""
    crash = rng.random(size) < 0.08
    return np.where(crash, rng.normal(-0.06, 0.04, size), rng.normal(0.012, 0.01, size))


def test_the_non_normal_standard_error_is_what_simulation_finds():
    rng = np.random.default_rng(1)
    big = _skewed(rng, 4_000_000)
    sr, sk, ku, _ = sharpe_moments(big)
    n = 400
    est = np.array([sharpe_moments(_skewed(rng, n))[0] for _ in range(6000)])
    corrected = sharpe_std_error(sr, n, sk, ku)
    naive = sharpe_std_error(sr, n)
    assert est.std() == pytest.approx(corrected, rel=0.06)
    assert abs(naive - est.std()) > 3 * abs(corrected - est.std())  # the correction is not cosmetic
    assert sk < -0.5 and ku > 4  # the scenario is genuinely non-normal


def test_expected_max_matches_simulation_and_is_zero_for_one_trial():
    assert expected_max_z(1) == 0.0
    rng = np.random.default_rng(2)
    for n_trials in (5, 24, 200):
        sim = rng.normal(size=(20000, n_trials)).max(axis=1).mean()
        assert expected_max_z(n_trials) == pytest.approx(sim, rel=0.03)
    assert expected_max_sharpe(24, 4.0) == pytest.approx(2 * expected_max_z(24))


def test_deflated_sharpe_has_the_right_size_and_some_power():
    """N zero-skill strategies: the naive PSR of the best rejects far too often; the DSR about 5 %."""
    rng = np.random.default_rng(3)
    n_strats, t, sims = 20, 500, 1500
    theory_var = 1.0 / (t - 1)
    naive_rej = dsr_rej = 0
    for _ in range(sims):
        x = rng.normal(0, 0.01, (t, n_strats))
        sr = x.mean(0) / x.std(0, ddof=1)
        best = int(np.argmax(sr))
        _, sk, ku, _ = sharpe_moments(x[:, best])
        out = deflated_sharpe(sr[best], t, sk, ku, n_strats, theory_var)
        naive_rej += out["psr_vs_zero"] > 0.95
        dsr_rej += out["dsr"] > 0.95
    assert naive_rej / sims > 0.4
    assert dsr_rej / sims < 0.09
    # power: one strategy with a true annualised Sharpe of 4 hides among the 19 nulls (two years of
    # data: a Sharpe of 2 would be detected only ~25 % of the time, by the same arithmetic)
    hits = 0
    for _ in range(300):
        x = rng.normal(0, 0.01, (t, n_strats))
        x[:, 0] += 0.01 * 4 / np.sqrt(252)
        sr = x.mean(0) / x.std(0, ddof=1)
        best = int(np.argmax(sr))
        _, sk, ku, _ = sharpe_moments(x[:, best])
        hits += deflated_sharpe(sr[best], t, sk, ku, n_strats, theory_var)["dsr"] > 0.95
    assert hits / 300 > 0.85


def test_min_track_record_and_min_backtest_length():
    n = min_track_record_length(0.1, 0.0, 0.0, 3.0, alpha=0.05)
    assert probabilistic_sharpe(0.1, 0.0, n, 0.0, 3.0) == pytest.approx(0.95, abs=1e-9)
    assert min_track_record_length(0.1, 0.1) == float("inf")
    assert min_track_record_length(0.1, 0.0, -1.0, 6.0) > min_track_record_length(0.1, 0.0)
    years = min_backtest_years(24, 1.0)
    assert years == pytest.approx(expected_max_z(24) ** 2)
    # after `years` years the expected best luck-only annual Sharpe equals the target
    assert expected_max_sharpe(24, 1.0 / (years * 252)) * np.sqrt(252) == pytest.approx(1.0)
    assert min_backtest_years(100, 1.0) > min_backtest_years(5, 1.0)


def test_effective_number_of_trials():
    rng = np.random.default_rng(4)
    one = rng.normal(size=(2000, 1))
    same = np.hstack([one] * 6)
    e = effective_trials(same)
    assert e["n_eff_average_correlation"] == pytest.approx(1.0) and e[
        "n_eff_participation_ratio"
    ] == pytest.approx(1.0)
    ind = effective_trials(rng.normal(size=(20000, 8)))
    assert ind["n_eff_average_correlation"] == pytest.approx(8.0, abs=0.15)
    assert ind["n_eff_participation_ratio"] == pytest.approx(8.0, abs=0.3)
    mixed = np.hstack([one + 0.5 * rng.normal(size=(2000, 1)) for _ in range(6)])
    m = effective_trials(mixed)
    assert 1.0 < m["n_eff_average_correlation"] < 6.0 and m[
        "mean_pairwise_correlation"
    ] == pytest.approx(0.8, abs=0.05)


def test_min_track_record_against_a_nonzero_benchmark_round_trips():
    for sr, star in ((0.12, 0.05), (0.2, -0.03)):
        n = min_track_record_length(sr, star, -0.5, 5.0, alpha=0.1)
        assert probabilistic_sharpe(sr, star, n, -0.5, 5.0) == pytest.approx(0.9, abs=1e-9)
    assert min_track_record_length(0.12, 0.05) > min_track_record_length(0.12, 0.0)  # a higher bar


def test_participation_ratio_differs_from_a_naive_count_on_unequal_clusters():
    rng = np.random.default_rng(12)
    a, b = rng.normal(size=(50000, 1)), rng.normal(size=(50000, 1))
    x = np.hstack([a, a, a, b])  # one cluster of 3 and a singleton: eigenvalues (3, 1, 0, 0)
    e = effective_trials(x)
    assert e["n_eff_participation_ratio"] == pytest.approx(16 / 10, abs=0.03)  # (sum l)^2 / sum l^2
    assert e["n_eff_average_correlation"] == pytest.approx(4 - 0.5 * 3, abs=0.03)  # rho = 0.5
