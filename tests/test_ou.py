"""OU estimation: exact recovery on noiseless paths, consistency, and the documented small-sample bias."""

from __future__ import annotations

import numpy as np
import pytest

from statarb.models.ou import LN2, fit_ou, half_life_from_b, simulate_ou


def test_exact_recovery_on_a_noiseless_ar1_path():
    """No noise -> the AR(1) fit is exact, so every OU parameter must come back to ~machine precision."""
    kappa, theta, dt = 0.05, 3.0, 1.0
    b = np.exp(-kappa * dt)
    x = theta + (10.0 - theta) * b ** np.arange(200)
    # a noiseless path makes the residual variance ~0, so only kappa/theta/half-life are meaningful
    fit = fit_ou(x, dt=dt)
    assert fit.kappa == pytest.approx(kappa, rel=1e-8)
    assert fit.theta == pytest.approx(theta, rel=1e-8)
    assert fit.half_life == pytest.approx(LN2 / kappa, rel=1e-8)
    assert fit.b == pytest.approx(b, rel=1e-10)


@pytest.mark.parametrize("b, expected", [(0.5, 1.0), (0.9, 6.5788), (0.99, 68.9676)])
def test_half_life_values(b, expected):
    assert half_life_from_b(b) == pytest.approx(expected, rel=1e-4)
    assert half_life_from_b(0.9, dt=0.5) == pytest.approx(0.5 * half_life_from_b(0.9))
    assert (
        np.isnan(half_life_from_b(1.0))
        and np.isnan(half_life_from_b(1.2))
        and np.isnan(half_life_from_b(0.0))
    )


def test_half_life_is_when_the_expected_deviation_halves():
    kappa = 0.1
    fit = fit_ou(simulate_ou(kappa, 0.0, 1.0, 5000, rng=np.random.default_rng(1)))
    assert np.exp(-fit.kappa * fit.half_life) == pytest.approx(
        0.5
    )  # by construction of ln 2 / kappa


def test_consistency_estimates_converge_as_n_grows():
    kappa, theta, sigma, dt = 0.08, -2.0, 0.6, 1.0
    errs = []
    for n in (300, 3000, 30000):
        fit = fit_ou(simulate_ou(kappa, theta, sigma, n, dt, rng=np.random.default_rng(n)), dt=dt)
        errs.append(abs(fit.kappa - kappa) / kappa)
    assert errs[-1] < 0.05 and errs[-1] < errs[0] + 1e-9
    assert fit.theta == pytest.approx(theta, abs=0.15) and fit.sigma == pytest.approx(
        sigma, rel=0.03
    )


def test_dt_scales_kappa_and_sigma_but_not_the_ar_coefficient_structure():
    """Same physical process sampled every dt=1/252 vs dt=1: kappa in 1/year units."""
    kappa, sigma = 12.0, 0.4  # annualised
    dt = 1 / 252
    x = simulate_ou(kappa, 0.0, sigma, 60000, dt, rng=np.random.default_rng(3))
    fit = fit_ou(x, dt=dt)
    assert fit.kappa == pytest.approx(kappa, rel=0.1) and fit.sigma == pytest.approx(
        sigma, rel=0.05
    )
    assert fit.half_life == pytest.approx(LN2 / kappa, rel=0.1)  # in years; ~14.6 trading days
    assert fit.stationary_sd == pytest.approx(sigma / np.sqrt(2 * kappa), rel=0.05)


def test_ols_is_biased_towards_faster_reversion_and_kendall_correction_helps():
    """Documented small-sample bias: kappa is over-estimated at n=100; the correction shrinks the bias."""
    rng = np.random.default_rng(11)
    kappa, n, reps = 0.03, 100, 800
    plain, corrected = [], []
    for _ in range(reps):
        x = simulate_ou(kappa, 0.0, 1.0, n, rng=rng)
        plain.append(fit_ou(x).b)
        corrected.append(fit_ou(x, bias_correct=True).b)
    b_true = np.exp(-kappa)
    bias_plain, bias_corr = np.mean(plain) - b_true, np.mean(corrected) - b_true
    assert bias_plain < -0.02  # AR(1) coefficient biased down => kappa biased up
    assert abs(bias_corr) < abs(bias_plain) / 2


def test_mle_agrees_with_ols_in_large_samples():
    x = simulate_ou(0.05, 1.0, 0.5, 4000, rng=np.random.default_rng(7))
    o, m = fit_ou(x, method="ols"), fit_ou(x, method="mle")
    assert m.method == "mle"
    assert m.kappa == pytest.approx(o.kappa, rel=0.02) and m.theta == pytest.approx(
        o.theta, abs=0.02
    )
    assert m.sigma == pytest.approx(o.sigma, rel=0.02)


def test_mle_likelihood_is_at_least_as_good_as_the_ols_starting_point():
    from statarb.models.ou import _neg_loglik

    x = simulate_ou(0.1, 0.0, 1.0, 300, rng=np.random.default_rng(5))
    o, m = fit_ou(x, method="ols"), fit_ou(x, method="mle")
    nll = lambda f: _neg_loglik([np.log(f.kappa), f.theta, np.log(f.sigma)], x, 1.0)  # noqa: E731
    assert nll(m) <= nll(o) + 1e-6


def test_a_random_walk_is_flagged_as_not_mean_reverting():
    x = np.cumsum(np.random.default_rng(2).normal(size=400))
    fit = fit_ou(x)
    if fit.b >= 1:  # b is a fluke-dependent estimate near 1; either way nothing must be invented
        assert not fit.mean_reverting and np.isnan(fit.kappa) and np.isnan(fit.half_life)
    trend = np.arange(300.0) + np.random.default_rng(0).normal(size=300) * 0.01
    bad = fit_ou(trend)
    assert not bad.mean_reverting and np.isnan(bad.kappa) and np.isnan(bad.stationary_sd)


def test_half_life_interval_is_asymmetric_and_can_be_unbounded():
    x = simulate_ou(
        0.01, 0.0, 1.0, 120, rng=np.random.default_rng(4)
    )  # very slow reversion, short sample
    fit = fit_ou(x)
    if fit.mean_reverting:
        lo, hi = fit.half_life_ci
        assert lo <= fit.half_life <= hi
        assert hi == np.inf or (hi - fit.half_life) > (fit.half_life - lo)  # skewed to the right
    fast = fit_ou(simulate_ou(0.5, 0.0, 1.0, 5000, rng=np.random.default_rng(6)))
    lo, hi = fast.half_life_ci
    assert lo < fast.half_life < hi < 3 * fast.half_life


def test_b_confidence_interval_covers_the_truth_at_roughly_the_nominal_rate():
    rng = np.random.default_rng(31)
    b_true, cover, reps = np.exp(-0.05), 0, 600
    for _ in range(reps):
        fit = fit_ou(simulate_ou(0.05, 0.0, 1.0, 400, rng=rng))
        cover += abs(fit.b - b_true) <= 1.96 * fit.b_se
    assert 0.90 <= cover / reps <= 0.99  # the AR(1) OLS bias pulls coverage slightly below 95 %


def test_simulate_ou_matches_the_stationary_distribution():
    x = simulate_ou(0.2, 5.0, 2.0, 200000, rng=np.random.default_rng(9))
    assert x.mean() == pytest.approx(5.0, abs=0.1)
    assert x.std() == pytest.approx(2.0 / np.sqrt(2 * 0.2), rel=0.03)
    assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(np.exp(-0.2), abs=0.01)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        fit_ou(np.arange(5.0))
    with pytest.raises(ValueError):
        fit_ou(np.r_[np.nan, np.arange(30.0)])
    with pytest.raises(ValueError):
        fit_ou(np.arange(30.0), method="bogus")
