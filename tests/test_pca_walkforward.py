"""PCA walk-forward: book P&L accounting, cost model consistency with the pair model, the placebo,
events, phase guards, and end-to-end look-ahead on the synthetic store."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from statarb.backtest.costs import (
    BookCostModel,
    CostConfig,
    MarketData,
    PairCostModel,
    build_market_data,
)
from statarb.backtest.pca_walkforward import (
    FoldSignals,
    PCAStrategyConfig,
    book_pnl,
    book_states,
    book_weights,
    fold_signals,
    placebo_permutation,
    prepare_pca_fold,
    run_fold,
    walk_forward_pca,
)
from statarb.backtest.walkforward import make_folds
from statarb.config import SplitConfig
from statarb.data.holdout import HoldoutViolation
from statarb.signals.residuals import ResidualConfig, residual_signals

from .wf_helpers import EVENT_DATE, SCREEN, SPLIT, build

SPLIT_CFG = SplitConfig(**SPLIT)
STRAT = PCAStrategyConfig(
    residual=ResidualConfig(n_factors=2, fit_window=250, refit_every=21, beta_window=40)
)
COST = CostConfig(capital=1e7)


# ---- accounting -------------------------------------------------------------------------------
def test_book_pnl_by_hand_with_drift_aware_trades():
    w = np.array([[0.10, -0.10], [0.10, -0.10], [0.0, 0.0]])
    r = np.array([[0.0, 0.0], [0.10, -0.05], [np.nan, 0.02]])
    pnl, trade, held = book_pnl(w, r)
    assert held.tolist() == [[0, 0], [0.10, -0.10], [0.10, -0.10]]
    assert pnl == pytest.approx([0.0, 0.10 * 0.10 + 0.10 * 0.05, -0.10 * 0.02])  # NaN return = 0
    assert trade[0].tolist() == [0.10, 0.10]  # opened at the close of day 0
    # day 1: held weights drift to 0.11 / -0.095; holding the same target trades the drift back
    assert trade[1] == pytest.approx([abs(0.10 - 0.11), abs(-0.10 + 0.095)])
    # flat at the close: the whole drifted book closes (a missing return drifts by zero)
    assert trade[2] == pytest.approx([0.10, 0.10 * 1.02])


def test_book_cost_model_reproduces_the_pair_cost_model_on_a_two_name_book():
    idx = pd.bdate_range("2020-01-01", periods=6)
    rng = np.random.default_rng(0)
    adv = pd.DataFrame({"Y": [3e7] * 6, "X": [8e8] * 6}, index=idx)
    sig = pd.DataFrame(rng.uniform(0.01, 0.02, (6, 2)), index=idx, columns=["Y", "X"])
    hs = pd.DataFrame({"Y": [2.5e-4] * 6, "X": [1e-4] * 6}, index=idx)
    cfg = CostConfig(capital=1e8, slots=20)
    pair = PairCostModel(MarketData(adv, sig, hs), cfg)
    frame = pd.DataFrame(
        {
            "trade_y": [1.0, 0, 0.1, 0, 0, 1.0],
            "trade_x": [0.7, 0, 0.07, 0, 0, 0.7],
            "w_y": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "w_x": [-0.7, -0.7, -0.7, -0.7, -0.7, 0.0],
        },
        index=idx,
    )
    frame["trade"] = frame["trade_y"] + frame["trade_x"]
    expect = pair(frame, "Y", "X")
    # the same trades as a book on total capital: pair units / slots
    trade = np.column_stack([frame["trade_y"], frame["trade_x"]]) / cfg.slots
    held = np.vstack([np.zeros((1, 2)), np.column_stack([frame["w_y"], frame["w_x"]])[:-1]])
    got = BookCostModel(cfg)(trade, held / cfg.slots, adv.to_numpy(), sig.to_numpy(), hs.to_numpy())
    for key in ("cost_spread", "cost_commission", "cost_impact", "cost_borrow", "cost"):
        assert got[key] == pytest.approx(expect[key].to_numpy() / cfg.slots, rel=1e-12), key
    assert got["participation"] == pytest.approx(expect["participation"].to_numpy(), rel=1e-12)


def test_book_cost_model_refuses_to_trade_where_inputs_are_undefined():
    z = np.zeros((2, 2))
    with pytest.raises(ValueError, match="undefined"):
        BookCostModel(CostConfig())(
            np.array([[0.01, 0], [0, 0]]),
            z,
            np.full((2, 2), np.nan),
            np.ones((2, 2)),
            np.ones((2, 2)),
        )


# ---- the synthetic world ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def world(tmp_path_factory, calendar):
    pipe = build(tmp_path_factory.mktemp("pcawf"), calendar)
    folds = make_folds(SPLIT_CFG, calendar, 2013, 2014, 2, "rolling")
    fd = prepare_pca_fold(pipe, folds[0], SCREEN)
    yield pipe, folds, fd
    pipe.close()


@pytest.fixture(scope="module")
def fs(world):
    return fold_signals(world[2], STRAT, COST)


def test_universe_is_the_pair_screens_eligible_set_and_data_is_bounded_by_test_end(world):
    _, folds, fd = world
    assert fd.ret.index[0] >= folds[0].train_start and fd.ret.index[-1] == folds[0].test_end
    assert fd.ret.index[fd.first] == folds[0].test_start
    assert set(fd.tickers) <= set(fd.ret.columns) and len(fd.tickers) == fd.n_eligible


def test_event_bars_are_missing_returns_and_names_are_flat_before_the_ex_date(world, fs):
    _, _, fd = world
    assert fd.event["EVY"].sum() == 1 and bool(fd.event.loc[EVENT_DATE, "EVY"])
    assert np.isnan(fd.ret.loc[EVENT_DATE, "EVY"])
    cfg = STRAT
    w = book_weights(fs, cfg)
    j = fd.tickers.index("EVY")
    dates = fs.sig.dates
    d_ev = dates.get_loc(pd.Timestamp(EVENT_DATE))
    assert fs.event_next[d_ev - 1, j] and not fs.event_next[d_ev, j]
    # flat at the close before the ex-date (only the hedge may touch it; never a signal position)
    st = book_states(fs, cfg)
    assert st[d_ev - 1, j] == 0
    # and unscored for the whole regression window afterwards
    assert np.isnan(fs.sig.score[d_ev : d_ev + 40, j]).all()
    assert w.shape == fs.sig.score.shape


def test_book_is_neutral_flat_at_the_end_and_p_and_l_is_causal(world, fs):
    w = book_weights(fs, STRAT)
    assert (w[-1] == 0).all()
    assert np.abs(w.sum(axis=1)).max() < 1e-10
    for d in np.flatnonzero(np.abs(w).sum(axis=1) > 0)[::20]:
        u = np.isfinite(fs.sig.beta[d]).all(axis=1)
        assert np.abs(fs.sig.beta[d][u].T @ w[d][u]).max() < 1e-10
    out = run_fold(fs, STRAT, COST)
    assert out["net"].equals(out["pnl"] - out["cost"])
    assert out["gross_exposure"].iloc[0] == 0.0  # nothing is held before the first close
    assert out["n_names"].iloc[0] == 0
    # position at the close of day d earns day d+1 only: zeroing every return after d keeps P&L <= d+1
    for d in (60, 120):
        cut = fs.ret.copy()
        cut[d + 2 :] = 0.0
        pnl_cut, _, _ = book_pnl(w, cut)
        assert np.allclose(pnl_cut[: d + 2], out["pnl"].to_numpy()[: d + 2])


def test_a_leak_in_the_future_prices_changes_nothing_before_it(tmp_path, calendar, world):
    """End to end: rebuild the store with everything after a date replaced by noise."""
    _, folds, fd = world
    cut = "2013-07-01"
    base = run_fold(fold_signals(fd, STRAT, COST), STRAT, COST)
    pipe2 = build(tmp_path, calendar, mutate_after=cut, mutate_seed=3)
    try:
        fd2 = prepare_pca_fold(pipe2, folds[0], SCREEN)
        moved = run_fold(fold_signals(fd2, STRAT, COST), STRAT, COST)
    finally:
        pipe2.close()
    before = base.index <= pd.Timestamp(cut)
    assert before.sum() > 100
    for col in ("pnl", "cost", "trade", "gross_exposure"):
        assert np.allclose(base.loc[before, col], moved.loc[before, col], atol=1e-12), col
    assert not np.allclose(base.loc[~before, "pnl"], moved.loc[~before, "pnl"])  # sensitivity


def test_a_relabelled_placebo_is_the_same_construction_on_other_stocks(world, fs):
    rng = np.random.default_rng(0)
    perm = placebo_permutation(fs, rng)
    assert sorted(perm) == list(range(len(perm))) and (perm != np.arange(len(perm))).any()
    base_states = book_states(fs, STRAT)
    placebo_states = book_states(fs, STRAT, perm)
    assert base_states.any() and not np.array_equal(placebo_states, base_states)
    assert np.array_equal(placebo_states[:, perm], base_states)  # same paths, other stocks
    w = book_weights(fs, STRAT, perm)
    assert np.abs(w.sum(axis=1)).max() < 1e-10 and (w[-1] == 0).all()
    ident = book_weights(fs, STRAT, np.arange(len(perm)))
    assert np.array_equal(ident, book_weights(fs, STRAT))


def test_the_event_mask_flattens_an_open_position_the_day_before_an_ex_date(fs):
    d, j = 30, 3  # before the 60-day time stop of a standing signal
    score = np.full_like(fs.sig.score, np.nan)
    score[:, j] = -2.0  # a standing "cheap" signal: long every day
    nxt = np.zeros_like(fs.event_next)
    nxt[d, j] = True  # an ex-date falls on day d + 1
    forced = replace(fs, sig=replace(fs.sig, score=score), event_next=nxt)
    st = book_states(forced, STRAT)[:, j]
    assert st[d - 1] == 1 and st[d] == 0 and st[d + 1] == 1
    assert (book_states(replace(forced, event_next=np.zeros_like(nxt)), STRAT)[:55, j] == 1).all()


def factor_world(phi: float, seed: int = 0, t: int = 700, n: int = 40) -> FoldSignals:
    """Returns = factors + the change in an AR(1) idiosyncratic level (phi < 1: mean-reverting)."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, [0.010, 0.006], (t, 2))
    b = rng.normal(0, 1, (n, 2))
    b[:, 0] = np.abs(b[:, 0]) + 0.5
    x = np.zeros((t, n))
    for j in range(1, t):
        x[j] = phi * x[j - 1] + rng.normal(0, 0.01, n)
    r = pd.DataFrame(
        f @ b.T + np.diff(x, axis=0, prepend=0.0), index=pd.bdate_range("2019-01-01", periods=t)
    )
    cfg = ResidualConfig(
        n_factors=2, fit_window=250, refit_every=21, beta_window=40, signal="reversal"
    )
    sig = residual_signals(r, 300, cfg)
    d = len(sig.dates)
    market = {
        k: np.full((d, n), v) for k, v in (("adv", 1e10), ("sigma", 0.01), ("half_spread", 1e-4))
    }
    return FoldSignals(
        None,
        sig,
        market,
        r.to_numpy()[300:],
        np.ones((d, n), dtype=bool),
        np.zeros((d, n), dtype=bool),
    )


def placebo_gross(fs_, n_draws=40, seed=1):
    rng = np.random.default_rng(seed)
    return np.array(
        [
            run_fold(fs_, STRAT, perm=placebo_permutation(fs_, rng))["pnl"].sum()
            for _ in range(n_draws)
        ]
    )


def test_mean_reverting_residuals_are_harvested_and_beat_the_relabelled_placebo():
    fs_ = factor_world(phi=0.5)
    real = run_fold(fs_, STRAT)["pnl"].sum()
    plc = placebo_gross(fs_)
    assert real > 0.05 and real > plc.max()
    assert (
        abs(plc.mean()) < 0.25 * real
    )  # the placebo is centred near zero, far below the real book


def test_placebo_is_calibrated_when_there_is_nothing_to_find():
    """False-positive control: with random-walk residuals the real book is one more draw from the
    placebo distribution -- ranks are spread over (0, 1) and the dispersions agree.  (Measured over 40
    seeds while building this: mean rank 0.56, placebo sd / real sd = 0.92; the bounds here are wide.)"""
    ranks, reals, plc_sd = [], [], []
    for seed in range(12):
        fs_ = factor_world(phi=1.0, seed=100 + seed)
        real = run_fold(fs_, STRAT)["pnl"].sum()
        plc = placebo_gross(fs_, n_draws=30, seed=seed)
        ranks.append((plc < real).mean())
        reals.append(real)
        plc_sd.append(plc.std())
    assert 0.3 < np.mean(ranks) < 0.7
    assert 0.6 < np.mean(plc_sd) / np.std(reals) < 1.6
    assert max(np.abs(reals)) < 0.1  # nowhere near the mean-reverting world's profit


def test_phase_and_holdout_guards(world, calendar):
    pipe, _, fd = world
    val = make_folds(SPLIT_CFG, calendar, 2014, 2014, 2, "rolling")[0]
    fd_val = prepare_pca_fold(pipe, val, SCREEN)
    with pytest.raises(ValueError, match="not allowed"):
        walk_forward_pca([fold_signals(fd_val, STRAT, COST)], STRAT, COST)
    walk_forward_pca(
        [fold_signals(fd_val, STRAT, COST)], STRAT, COST, allowed_phases=("validation",)
    )
    hold = make_folds(SPLIT_CFG, calendar, 2015, 2015, 2, "rolling")[0]
    with pytest.raises(HoldoutViolation):
        prepare_pca_fold(pipe, hold, SCREEN)
    assert fd is not None


def test_costs_reduce_net_and_components_are_consistent(fs):
    gross = run_fold(fs, STRAT)
    net = run_fold(fs, STRAT, COST)
    assert (gross["cost"] == 0).all() and np.allclose(gross["pnl"], net["pnl"])
    comps = net[["cost_spread", "cost_commission", "cost_impact", "cost_borrow"]].sum(axis=1)
    assert np.allclose(comps, net["cost"]) and (net["cost"] >= 0).all()
    pricey = run_fold(fs, STRAT, CostConfig(capital=1e9))
    assert (
        pricey["cost_impact"].sum() > 3 * net["cost_impact"].sum()
    )  # impact is super-linear in size
    assert build_market_data is not None


def test_the_no_factor_control_is_dollar_neutral_and_reports_its_signal_names(world):
    cfg0 = PCAStrategyConfig(
        residual=ResidualConfig(n_factors=0, fit_window=250, beta_window=40, signal="reversal")
    )
    fs0 = fold_signals(world[2], cfg0, COST)
    out = run_fold(fs0, cfg0, COST)
    assert out["net_exposure"].abs().max() < 1e-10 and out["gross_exposure"].max() > 0
    assert (out["n_signal"] <= out["n_names"]).all() and out["n_signal"].max() > 0
    assert out["n_signal"].iloc[0] == 0  # nothing is held before the first close
