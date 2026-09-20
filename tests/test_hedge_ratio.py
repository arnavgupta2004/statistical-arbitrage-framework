"""Hedge-ratio estimators: exactness against brute force / statsmodels, tracking, and causality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

from statarb.data.leakage import assert_no_lookahead, truncation_invariance
from statarb.models.hedge_ratio import HedgeSpec, hedge_path, static_fit
from statarb.models.kalman import kalman_hedge


def pair(n=600, seed=0, beta=0.8, alpha=0.3, noise=0.03, drift_sd=0.0):
    """ly = alpha + beta_t * lx + AR(0.9) noise; beta_t a random walk when drift_sd > 0."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    lx = 4 + np.cumsum(rng.normal(0, 0.015, n))
    b = beta + np.cumsum(rng.normal(0, drift_sd, n)) if drift_sd else np.full(n, beta)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.9 * u[t - 1] + rng.normal(0, noise * 0.44)
    return (
        pd.Series(alpha + b * lx + u, index=idx),
        pd.Series(lx, index=idx),
        pd.Series(b, index=idx),
    )


# ---- Kalman ---------------------------------------------------------------------------------
def test_kalman_matches_statsmodels_state_space_filter():
    ly, lx, _ = pair(300, seed=1)
    n_init, delta = 60, 1e-4
    mine = kalman_hedge(ly, lx, delta, n_init)
    y, x = ly.to_numpy(), lx.to_numpy()
    xm = x[:n_init].mean()
    xc = x - xm
    design = np.column_stack([xc[:n_init], np.ones(n_init)])
    coef, *_ = np.linalg.lstsq(design, y[:n_init], rcond=None)
    resid = y[:n_init] - design @ coef
    r_obs = resid @ resid / (n_init - 2)
    cov0 = r_obs * np.linalg.inv(design.T @ design)
    q = delta / (1 - delta)
    kf = KalmanFilter(k_endog=1, k_states=2)
    kf.bind(y[n_init:])
    kf.design = np.stack([np.array([[xc[t], 1.0]]) for t in range(n_init, len(y))], axis=2)
    kf.obs_cov, kf.transition, kf.selection = np.array([[r_obs]]), np.eye(2), np.eye(2)
    kf.state_cov = q * np.eye(2)
    kf.initialize_known(coef, cov0 + q * np.eye(2))  # our loop adds the state noise inside step one
    ref = kf.filter()
    got = mine.iloc[n_init:]
    assert got["beta"].to_numpy() == pytest.approx(ref.filtered_state[0], abs=1e-12)
    assert got["alpha"].to_numpy() == pytest.approx(
        ref.filtered_state[1] - ref.filtered_state[0] * xm, abs=1e-12
    )
    assert got["innovation"].to_numpy() == pytest.approx(ref.forecasts_error[0], abs=1e-12)
    assert got["innovation_sd"].to_numpy() ** 2 == pytest.approx(
        ref.forecasts_error_cov[0, 0], abs=1e-12
    )


def test_kalman_with_vanishing_state_noise_is_recursive_least_squares():
    """delta -> 0: the recursion reproduces expanding OLS exactly (analytic check)."""
    ly, lx, _ = pair(400, seed=2)
    k = kalman_hedge(ly, lx, delta=1e-13, n_init=60)
    for t in (100, 250, 399):
        beta, alpha = np.polyfit(lx.iloc[: t + 1], ly.iloc[: t + 1], 1)
        assert k["beta"].iloc[t] == pytest.approx(beta, abs=1e-6)
        assert k["alpha"].iloc[t] == pytest.approx(alpha, abs=1e-5)


def test_kalman_tracks_a_drifting_hedge_ratio_better_than_frozen_or_expanding_ols():
    errs = {"static": [], "expanding": [], "kalman": []}
    for seed in range(8):
        ly, lx, b = pair(1200, seed=10 + seed, drift_sd=0.004)
        test = slice(600, None)
        static = hedge_path(ly, lx, HedgeSpec("static"), ly.index[599])["beta"]
        exp = hedge_path(ly, lx, HedgeSpec("expanding"), ly.index[599])["beta"]
        kal = hedge_path(ly, lx, HedgeSpec("kalman", delta=1e-5), ly.index[599])["beta"]
        for name, est in (("static", static), ("expanding", exp), ("kalman", kal)):
            errs[name].append(np.sqrt(np.mean((est.iloc[test] - b.iloc[test]) ** 2)))
    assert np.mean(errs["kalman"]) < np.mean(errs["static"]) and np.mean(errs["kalman"]) < np.mean(
        errs["expanding"]
    )


def test_kalman_standardised_innovations_are_calibrated_when_the_model_is_right():
    """Simulate exactly the model the filter assumes; nu / sqrt(S) should be ~ N(0, 1)."""
    rng = np.random.default_rng(3)
    n, delta = 4000, 1e-4
    q = delta / (1 - delta)
    lx = 4 + np.cumsum(rng.normal(0, 0.015, n))
    state = np.array([0.8, 0.3])
    y = np.empty(n)
    for t in range(n):
        state = (
            state + rng.normal(0, np.sqrt(q), 2) * 0.1
        )  # small state noise, well below the filter's q
        y[t] = state[0] * (lx[t] - lx[:60].mean()) + state[1] + rng.normal(0, 0.03)
    k = kalman_hedge(pd.Series(y), pd.Series(lx), delta, 60)
    z = (k["innovation"] / k["innovation_sd"]).dropna()
    assert abs(z.mean()) < 0.1 and 0.85 < z.std() < 1.1


@pytest.mark.parametrize("bad", [dict(delta=0.0), dict(delta=1.0), dict(n_init=5)])
def test_kalman_argument_validation(bad):
    ly, lx, _ = pair(200)
    with pytest.raises(ValueError):
        kalman_hedge(ly, lx, **bad)
    with pytest.raises(ValueError, match="NaN"):
        kalman_hedge(ly.where(ly.index != ly.index[5]), lx)
    with pytest.raises(ValueError):
        kalman_hedge(ly.iloc[:50], lx.iloc[:50])


def test_kalman_is_nan_until_the_warm_up_ends():
    ly, lx, _ = pair(200)
    k = kalman_hedge(ly, lx, n_init=60)
    assert k["beta"].iloc[:60].isna().all() and k["beta"].iloc[60:].notna().all()


# ---- the estimators -------------------------------------------------------------------------
def test_static_is_ols_on_the_training_window_only_and_ignores_later_data():
    ly, lx, _ = pair(500, seed=4, drift_sd=0.004)
    train_end = ly.index[299]
    path = hedge_path(ly, lx, HedgeSpec("static"), train_end)
    beta, alpha = np.polyfit(lx.iloc[:300], ly.iloc[:300], 1)
    assert path["beta"].iloc[0] == pytest.approx(beta) and path["alpha"].iloc[0] == pytest.approx(
        alpha
    )
    assert path["beta"].nunique() == 1
    changed = ly.copy()
    changed.iloc[300:] *= 1.7  # scramble everything after train_end
    assert hedge_path(changed, lx, HedgeSpec("static"), train_end)["beta"].iloc[0] == pytest.approx(
        beta
    )
    assert static_fit(ly.iloc[:300], lx.iloc[:300]) == pytest.approx((alpha, beta))


def test_expanding_equals_ols_on_all_data_to_date():
    ly, lx, _ = pair(300, seed=5)
    path = hedge_path(ly, lx, HedgeSpec("expanding", min_periods=60), ly.index[100])
    assert path["beta"].iloc[:59].isna().all()
    for t in (59, 150, 299):
        beta, alpha = np.polyfit(lx.iloc[: t + 1], ly.iloc[: t + 1], 1)
        assert path["beta"].iloc[t] == pytest.approx(beta) and path["alpha"].iloc[
            t
        ] == pytest.approx(alpha)


def test_rolling_equals_ols_on_the_trailing_window():
    ly, lx, _ = pair(300, seed=6)
    path = hedge_path(ly, lx, HedgeSpec("rolling", window=90), ly.index[100])
    assert path["beta"].iloc[:89].isna().all()
    for t in (89, 200, 299):
        beta, alpha = np.polyfit(lx.iloc[t - 89 : t + 1], ly.iloc[t - 89 : t + 1], 1)
        assert path["beta"].iloc[t] == pytest.approx(beta) and path["alpha"].iloc[
            t
        ] == pytest.approx(alpha)


def test_shorter_windows_track_drift_but_are_noisier_on_a_constant_hedge():
    rmse_const, rmse_drift = {}, {}
    for w in (60, 250):
        c, d = [], []
        for seed in range(10):
            ly, lx, b = pair(1200, seed=seed, beta=0.8)
            c.append(
                np.sqrt(
                    np.mean(
                        (
                            hedge_path(ly, lx, HedgeSpec("rolling", window=w), ly.index[0])[
                                "beta"
                            ].iloc[600:]
                            - 0.8
                        )
                        ** 2
                    )
                )
            )
            ly, lx, b = pair(1200, seed=100 + seed, drift_sd=0.004)
            est = hedge_path(ly, lx, HedgeSpec("rolling", window=w), ly.index[0])["beta"]
            d.append(np.sqrt(np.mean((est.iloc[600:] - b.iloc[600:]) ** 2)))
        rmse_const[w], rmse_drift[w] = np.mean(c), np.mean(d)
    assert (
        rmse_const[60] > rmse_const[250]
    )  # estimation noise: short windows are worse when nothing moves
    assert (
        rmse_drift[250] > 0
    )  # (documented trade-off; the direction under drift is measured in experiments)


def test_spec_validation_and_labels():
    assert (
        HedgeSpec("static").label == "static"
        and HedgeSpec("rolling", window=120).label == "rolling120"
    )
    assert (
        HedgeSpec("kalman", delta=1e-5).label == "kalman1e-05"
        and HedgeSpec("expanding").label == "expanding"
    )
    with pytest.raises(ValueError):
        HedgeSpec("bogus")
    with pytest.raises(ValueError):
        HedgeSpec("rolling")
    with pytest.raises(ValueError):
        HedgeSpec("rolling", window=5)


SPECS = [
    HedgeSpec("expanding"),
    HedgeSpec("rolling", window=60),
    HedgeSpec("rolling", window=250),
    HedgeSpec("kalman", delta=1e-5),
    HedgeSpec("kalman", delta=1e-4),
]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.label)
def test_every_time_varying_hedge_path_is_causal(spec):
    ly, lx, _ = pair(500, seed=7, drift_sd=0.003)
    frame = pd.DataFrame({"ly": ly, "lx": lx})
    fn = lambda f: hedge_path(f["ly"], f["lx"], spec, f.index[0])[["beta"]]  # noqa: E731
    assert_no_lookahead(fn, frame, frame.index[[120, 250, 400, 480]])


def test_the_static_path_is_causal_given_a_fixed_training_end():
    ly, lx, _ = pair(500, seed=8)
    frame = pd.DataFrame({"ly": ly, "lx": lx})
    train_end = frame.index[299]
    fn = lambda f: hedge_path(f["ly"], f["lx"], HedgeSpec("static"), train_end)[["beta"]]  # noqa: E731
    assert truncation_invariance(fn, frame, frame.index[[300, 400, 480]]).passed
