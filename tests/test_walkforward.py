"""Walk-forward engine: folds, phases, slots, forced liquidation, events, costs and, above all,
look-ahead (selection and trading must not depend on anything after the dates they may use)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from statarb.backtest.walkforward import (
    FoldData,
    StrategyConfig,
    aggregate,
    blackout_mask,
    make_folds,
    prepare_fold,
    run_pairs,
    walk_forward,
)
from statarb.config import SplitConfig
from statarb.data.holdout import HoldoutViolation

from .wf_helpers import EVENT_DATE, SCREEN, SPLIT, build

STRAT = StrategyConfig()
SPLIT_CFG = SplitConfig(**SPLIT)


@pytest.fixture(scope="module")
def world(tmp_path_factory, calendar):
    pipe = build(tmp_path_factory.mktemp("wf"), calendar)
    folds = make_folds(SPLIT_CFG, calendar, 2013, 2014, 2, "rolling")
    fds = [prepare_fold(pipe, f, SCREEN) for f in folds]
    yield pipe, folds, fds
    pipe.close()


@pytest.fixture(scope="module")
def research(world):
    return world[2][0]


# ---- folds ------------------------------------------------------------------------------------
def test_folds_are_chronological_with_a_gap_of_zero_between_train_and_test(calendar):
    folds = make_folds(SPLIT_CFG, calendar, 2013, 2015, 2, "rolling")
    assert [f.phase for f in folds] == ["research", "validation", "holdout"]
    for f in folds:
        assert f.train_start < f.train_end < f.test_start <= f.test_end
        nxt = calendar.sessions(f.train_end + pd.Timedelta(days=1), f.test_start)[0]
        assert nxt == f.test_start  # the training window ends on the session right before the block
    assert folds[0].test_start.year == folds[0].test_end.year == 2013
    assert folds[1].test_start > folds[0].test_end
    assert folds[0].train_start == pd.Timestamp("2011-01-03")  # rolling 2y, clipped at data_start


def test_rolling_and_expanding_training_windows(calendar):
    roll = make_folds(SPLIT_CFG, calendar, 2013, 2014, 2, "rolling")
    exp = make_folds(SPLIT_CFG, calendar, 2013, 2014, 2, "expanding")
    assert roll[1].train_start > pd.Timestamp("2011-12-01")  # about two years before 2014
    assert exp[1].train_start == pd.Timestamp("2011-01-03")  # everything since data_start
    assert exp[1].train_end == roll[1].train_end


def test_fold_construction_errors(calendar):
    with pytest.raises(ValueError, match="fewer than 250"):
        make_folds(SPLIT_CFG, calendar, 2011, 2011, 2)
    straddle = SplitConfig(
        **{
            **SPLIT,
            "research_end": pd.Timestamp("2013-06-30").date(),
            "validation_start": pd.Timestamp("2013-07-01").date(),
        }
    )
    with pytest.raises(ValueError, match="straddles"):
        make_folds(straddle, calendar, 2013, 2013, 2)
    with pytest.raises(ValueError):
        make_folds(SPLIT_CFG, calendar, 2013, 2013, 2, mode="anchored")


def test_data_end_truncates_the_last_block(calendar):
    f = make_folds(SPLIT_CFG, calendar, 2013, 2013, 2, data_end="2013-09-30")[0]
    assert f.test_end == pd.Timestamp("2013-09-30")


# ---- phases and the holdout guard --------------------------------------------------------------
def test_walk_forward_refuses_phases_it_was_not_told_to_run(world):
    _, _, fds = world
    with pytest.raises(ValueError, match="validation"):
        walk_forward(fds, STRAT)  # fds includes a validation fold; default allows research only
    assert len(walk_forward(fds[:1], STRAT).daily) > 0
    assert len(walk_forward(fds, STRAT, allowed_phases=("research", "validation")).daily) > 0


def test_a_holdout_fold_cannot_even_be_prepared_until_the_holdout_is_unlocked(world, calendar):
    pipe, _, _ = world
    fold = make_folds(SPLIT_CFG, calendar, 2015, 2015, 2)[0]
    assert fold.phase == "holdout"
    with pytest.raises(HoldoutViolation):
        prepare_fold(pipe, fold, SCREEN)


# ---- the run itself ------------------------------------------------------------------------------
def test_screen_finds_the_planted_pairs_and_the_run_earns_a_gross_return(research):
    sel = research.screen.selected
    names = {frozenset((r.y, r.x)) for r in sel.itertuples()}
    assert sum(frozenset((f"Y{k}", f"X{k}")) in names for k in range(4)) >= 3
    res = walk_forward([research], STRAT)
    assert res.daily["pnl"].sum() > 0 and res.daily["cost"].sum() == 0 and len(res.trades) > 0
    assert res.daily.index.equals(
        research.logp.loc[research.fold.test_start : research.fold.test_end].index
    )
    assert res.folds.loc[0, "phase"] == "research" and res.folds.loc[0, "n_selected"] == len(sel)


def test_portfolio_return_is_the_slot_weighted_sum_of_pair_returns(research):
    pairs = list(zip(research.screen.selected["y"], research.screen.selected["x"], strict=True))
    runs = run_pairs(research, STRAT, pairs)
    idx = runs[0].frame.index
    manual = sum(r.frame["pnl"] for r in runs) / STRAT.slots
    pd.testing.assert_series_equal(
        aggregate(runs, idx, STRAT.slots)["pnl"], manual, check_names=False
    )
    half = walk_forward([research], StrategyConfig(slots=2 * STRAT.slots)).daily["pnl"]
    assert half.sum() == pytest.approx(walk_forward([research], STRAT).daily["pnl"].sum() / 2)


def test_positions_are_forced_flat_at_the_end_of_the_block_and_the_exit_is_counted(research):
    pairs = list(zip(research.screen.selected["y"], research.screen.selected["x"], strict=True))
    runs = run_pairs(research, STRAT, pairs)
    for r in runs:
        assert r.frame["w_y"].iloc[-1] == 0 and r.frame["w_x"].iloc[-1] == 0
    # a pair that was open the day before must show the liquidation trade on the last bar
    open_before = [r for r in runs if r.frame["w_y"].iloc[-2] != 0]
    for r in open_before:
        assert r.frame["trade"].iloc[-1] > 0
        assert r.trades["exit_reason"].iloc[-1] == "end"


def test_blocks_are_independent_a_fold_gives_the_same_result_alone_or_with_others(world):
    _, _, fds = world
    alone = walk_forward(fds[:1], STRAT).daily
    together = walk_forward(fds, STRAT, allowed_phases=("research", "validation")).daily
    pd.testing.assert_frame_equal(alone, together[together["fold"] == 0])


def test_the_run_is_deterministic(research):
    pd.testing.assert_frame_equal(
        walk_forward([research], STRAT).daily, walk_forward([research], STRAT).daily
    )


def test_costs_are_a_hook_that_leaves_gross_untouched(research):
    base = walk_forward([research], STRAT).daily
    cost = walk_forward([research], STRAT, cost_fn=lambda f: 0.001 * f["trade"]).daily
    assert cost["pnl"].sum() == pytest.approx(base["pnl"].sum())
    assert cost["cost"].sum() == pytest.approx(0.001 * base["trade"].sum())
    assert cost["net"].sum() == pytest.approx(base["pnl"].sum() - 0.001 * base["trade"].sum())


def test_random_selector_is_reproducible_matches_the_count_and_avoids_significant_pairs(research):
    a = walk_forward([research], STRAT, "random", np.random.default_rng(3))
    b = walk_forward([research], STRAT, "random", np.random.default_rng(3))
    c = walk_forward([research], STRAT, "random", np.random.default_rng(4))
    pd.testing.assert_frame_equal(a.daily, b.daily)
    assert not np.allclose(a.daily["pnl"], c.daily["pnl"])
    assert a.folds.loc[0, "n_selected"] == len(research.screen.selected)
    # the pairs CHOSEN (not just those that happened to trade): none significant, all beta > 0
    pairs = research.screen.pairs.set_index(["y", "x"])
    for seed in range(15):
        chosen = walk_forward([research], STRAT, "random", np.random.default_rng(seed)).folds.loc[
            0, "pairs"
        ]
        assert len(chosen) == len(research.screen.selected)
        assert (pairs.loc[chosen, "p_calibrated"] > 0.05).all() and (
            pairs.loc[chosen, "beta"] > 0
        ).all()
    with pytest.raises(ValueError, match="rng"):
        walk_forward([research], STRAT, "random")


# ---- events --------------------------------------------------------------------------------------
def test_blackout_mask_geometry():
    ev = np.zeros(20, dtype=bool)
    ev[[5, 18]] = True
    m = blackout_mask(ev, lead=1, after=3)
    assert np.flatnonzero(m).tolist() == [4, 5, 6, 7, 8, 17, 18, 19]  # clipped at the end
    assert not blackout_mask(np.zeros(5, dtype=bool), 1, 3).any()


def test_a_pair_is_kept_flat_around_a_non_ordinary_distribution(research):
    ev = pd.Timestamp(EVENT_DATE)
    assert bool(research.event["EVY"].loc[ev]) and not research.event["EVX"].any()
    blocked = blackout_mask(research.event["EVY"].to_numpy(), 1, STRAT.z_window)
    days = research.logp.index[blocked]
    days = days[(days >= research.fold.test_start) & (days <= research.fold.test_end)]
    assert (
        len(days) > STRAT.z_window and days[0] < ev
    )  # starts before the ex-date, lasts z_window bars
    safe = run_pairs(research, STRAT, [("EVY", "EVX")])[0].frame
    assert (safe.loc[days, "gross"] == 0).all()  # flat from the day before the ex-date, for 60 bars
    # with events switched off the same pair trades through the window and takes the +40 % "return"
    fd2 = FoldData(
        research.fold,
        research.screen,
        research.logp,
        research.ret,
        pd.DataFrame(False, index=research.logp.index, columns=research.logp.columns),
    )
    unsafe = run_pairs(fd2, StrategyConfig(), [("EVY", "EVX")])[0].frame
    assert unsafe.loc[days, "gross"].sum() > 0


# ---- look-ahead: the engine as a whole -------------------------------------------------------
def result_key(fd):
    return walk_forward([fd], STRAT).daily


def test_nothing_after_the_test_block_matters(world, tmp_path, calendar):
    """Scramble every price after test_end: selection, positions and P&L must be identical."""
    _, _, fds = world
    noisy = build(tmp_path / "noisy", calendar, mutate_after="2013-12-31", mutate_seed=9)
    try:
        folds = make_folds(SPLIT_CFG, calendar, 2013, 2013, 2)
        other = prepare_fold(noisy, folds[0], SCREEN)
    finally:
        noisy.close()
    pd.testing.assert_frame_equal(fds[0].screen.pairs, other.screen.pairs)
    pd.testing.assert_frame_equal(result_key(fds[0]), result_key(other))


def test_selection_does_not_depend_on_the_test_block_at_all(world, tmp_path, calendar):
    """Scramble every price from the first test day on: the screen (training data only) is unchanged."""
    _, _, fds = world
    noisy = build(tmp_path / "noisy2", calendar, mutate_after="2012-12-31", mutate_seed=4)
    try:
        other = prepare_fold(noisy, make_folds(SPLIT_CFG, calendar, 2013, 2013, 2)[0], SCREEN)
    finally:
        noisy.close()
    pd.testing.assert_frame_equal(fds[0].screen.pairs, other.screen.pairs)
    assert not np.allclose(
        result_key(fds[0])["pnl"], result_key(other)["pnl"]
    )  # trading does see it


def test_pnl_up_to_day_t_does_not_depend_on_prices_after_t(world, tmp_path, calendar):
    """Scramble prices after a date inside the block: P&L on or before that date is identical."""
    _, _, fds = world
    cut = pd.Timestamp("2013-08-30")
    noisy = build(tmp_path / "noisy3", calendar, mutate_after=cut, mutate_seed=2)
    try:
        other = prepare_fold(noisy, make_folds(SPLIT_CFG, calendar, 2013, 2013, 2)[0], SCREEN)
    finally:
        noisy.close()
    a, b = result_key(fds[0]), result_key(other)
    pd.testing.assert_frame_equal(
        a.loc[:cut, ["pnl", "trade", "gross"]], b.loc[:cut, ["pnl", "trade", "gross"]]
    )
    assert not np.allclose(
        a.loc[cut + pd.Timedelta(days=5) :, "pnl"], b.loc[cut + pd.Timedelta(days=5) :, "pnl"]
    )


def test_a_store_that_ends_at_test_end_gives_the_same_fold(world, tmp_path, calendar):
    _, _, fds = world
    short = build(tmp_path / "short", calendar, truncate_after="2013-12-31")
    try:
        other = prepare_fold(short, make_folds(SPLIT_CFG, calendar, 2013, 2013, 2)[0], SCREEN)
    finally:
        short.close()
    pd.testing.assert_frame_equal(fds[0].screen.pairs, other.screen.pairs)
    pd.testing.assert_frame_equal(result_key(fds[0]), result_key(other))


def test_strategy_config_is_serialisable_and_rejects_unknown_fields():
    d = StrategyConfig(hedge_method="rolling", hedge_window=120).model_dump()
    assert d["hedge_method"] == "rolling" and StrategyConfig(**d).hedge.label == "rolling120"
    assert (
        StrategyConfig().label == "static_in2" and StrategyConfig(entry=1.5).label == "static_in1.5"
    )
    with pytest.raises(ValidationError):
        StrategyConfig(entri=2.0)
