"""Cost model: hand-worked arithmetic, scaling laws, Corwin-Schultz, causality, engine integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.backtest.costs import (
    K_CS,
    CostConfig,
    MarketData,
    PairCostModel,
    breakeven_multiple,
    build_market_data,
    corwin_schultz,
    tier_half_spread,
)
from statarb.backtest.walkforward import (
    StrategyConfig,
    aggregate,
    make_folds,
    prepare_fold,
    run_pairs,
    walk_forward,
)
from statarb.config import SplitConfig
from statarb.data.leakage import assert_no_lookahead

from .wf_helpers import SCREEN, SPLIT, build

# ---- Corwin-Schultz ------------------------------------------------------------------------------


def frame(**cols):
    idx = pd.bdate_range("2020-01-01", periods=len(next(iter(cols.values()))))
    return pd.DataFrame({"A": next(iter(cols.values()))}, index=idx), idx


def test_corwin_schultz_by_hand():
    idx = pd.bdate_range("2020-01-01", periods=2)
    high = pd.DataFrame({"A": [101.0, 102.0]}, index=idx)
    low = pd.DataFrame({"A": [99.0, 100.0]}, index=idx)
    est = corwin_schultz(high, low)["A"]
    assert np.isnan(est.iloc[0])  # needs two days
    beta = np.log(101 / 99) ** 2 + np.log(102 / 100) ** 2
    gamma = np.log(102 / 99) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / K_CS - np.sqrt(gamma / K_CS)
    assert est.iloc[1] == pytest.approx(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)))
    assert K_CS == pytest.approx(3 - 2 * np.sqrt(2))


def test_the_overnight_adjustment_shifts_the_range_by_the_gap():
    idx = pd.bdate_range("2020-01-01", periods=3)
    high = pd.DataFrame({"A": [101.0, 101.0, 104.0]}, index=idx)
    low = pd.DataFrame({"A": [99.0, 99.0, 102.0]}, index=idx)
    close = pd.DataFrame(
        {"A": [100.0, 100.0, 100.0]}, index=idx
    )  # today's range is entirely above the prior close
    plain = corwin_schultz(high, low)["A"].iloc[2]
    adj = corwin_schultz(high, low, close)["A"].iloc[2]
    shifted_h, shifted_l = (
        104.0 - 2.0,
        102.0 - 2.0,
    )  # low - prior close = 2 -> shift the day down by 2
    manual = corwin_schultz(
        pd.DataFrame({"A": [101.0, 101.0, shifted_h]}, index=idx),
        pd.DataFrame({"A": [99.0, 99.0, shifted_l]}, index=idx),
    )["A"].iloc[2]
    assert adj == pytest.approx(manual) and adj != pytest.approx(plain)


def _simulated_bars(spread, n=3000, sigma=0.015, steps=390, seed=0):
    rng = np.random.default_rng(seed)
    mid = 100 * np.exp(
        np.cumsum(rng.normal(0, sigma / np.sqrt(steps), (n, steps)).reshape(-1)).reshape(n, steps)
    )
    idx = pd.bdate_range("2000-01-01", periods=n)
    mk = lambda v: pd.DataFrame({"A": v}, index=idx)  # noqa: E731
    return mk(mid.max(1) * (1 + spread / 2)), mk(mid.min(1) * (1 - spread / 2)), mk(mid[:, -1])


@pytest.mark.parametrize("spread, tol", [(0.01, 0.0008), (0.005, 0.0012)])
def test_corwin_schultz_recovers_a_wide_spread_on_simulated_bars(spread, tol):
    h, lo, c = _simulated_bars(spread)
    assert corwin_schultz(h, lo, c)["A"].mean() == pytest.approx(spread, abs=tol)


def test_corwin_schultz_is_biased_up_and_noisy_for_a_tiny_spread_the_reason_it_is_not_the_central_model():
    """True spread 0: the estimate averages several bps and is often far larger or negative."""
    h, lo, c = _simulated_bars(0.0)
    est = corwin_schultz(h, lo, c)["A"].dropna()
    assert 0.0002 < est.mean() < 0.001  # a few bps, not zero (documented upward bias)
    assert est.std() > 0.005 and (est < 0).mean() > 0.1  # very noisy, often negative


# ---- tiers and market data -----------------------------------------------------------------------


def test_tier_half_spread_boundaries():
    adv = pd.DataFrame({"A": [6e8, 5e8, 2e8, 1e8, 5e7, 2.5e7, 1e6, np.nan]})
    got = tier_half_spread(adv)["A"].to_numpy()
    assert got[:7] == pytest.approx(np.array([1.0, 1.0, 1.5, 1.5, 2.5, 2.5, 5.0]) / 1e4)
    assert np.isnan(got[7])


def market(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.012, (n, 2)), axis=0)), index=idx, columns=list("AB")
    )
    high, low = close * 1.01, close * 0.99
    dv = pd.DataFrame(np.exp(rng.normal(np.log(2e8), 0.3, (n, 2))), index=idx, columns=list("AB"))
    return dv, close.pct_change(), high, low, close


@pytest.mark.parametrize("model", ["tiered", "corwin_schultz", "max"])
def test_market_data_is_causal_for_every_spread_model(model):
    dv, ret, high, low, close = market()
    cfg = CostConfig(spread_model=model)
    stacked = pd.concat({"dv": dv, "ret": ret, "high": high, "low": low, "close": close}, axis=1)

    def fn(f):
        m = build_market_data(f["dv"], f["ret"], f["high"], f["low"], f["close"], cfg)
        return pd.concat({"adv": m.adv, "sigma": m.sigma, "half": m.half_spread}, axis=1)

    assert_no_lookahead(fn, stacked, stacked.index[[80, 150, 230, 290]], atol=1e-10)


def test_market_data_uses_yesterdays_volume_not_todays():
    """Plant an enormous volume today and a tiny one at the far edge of the window: a median that
    (wrongly) included today would move; swapping one ordinary value for another would not."""
    dv, ret, high, low, close = market()
    t = 100
    dv = dv.copy()
    dv.iloc[t, 0], dv.iloc[t - 20, 0] = 1e13, 1.0
    ret = ret.copy()
    ret.iloc[t, 0] = 0.9  # a huge return today
    m = build_market_data(dv, ret, high, low, close, CostConfig())
    assert m.adv["A"].iloc[t] == pytest.approx(
        dv["A"].iloc[t - 20 : t].median()
    )  # window ends at t-1
    assert m.adv["A"].iloc[t] != pytest.approx(dv["A"].iloc[t - 19 : t + 1].median())
    assert m.sigma["A"].iloc[t] == pytest.approx(ret["A"].iloc[t - 20 : t].std())
    assert m.sigma["A"].iloc[t] < 0.1  # today's 90 % return is not in it yet


# ---- the cost arithmetic -------------------------------------------------------------------------


def hand_frame():
    idx = pd.bdate_range("2020-01-01", periods=4)
    return pd.DataFrame(
        {
            "trade_y": [0.0, 1.0, 0.0, 1.0],
            "trade_x": [0.0, 0.5, 0.0, 0.5],
            "w_y": [0.0, 1.0, 1.0, 0.0],
            "w_x": [0.0, -0.5, -0.5, 0.0],
        },
        index=idx,
    ).assign(trade=lambda d: d["trade_y"] + d["trade_x"])


def const_market(index, adv=1e8, sigma=0.02, half=0.0002):
    mk = lambda v: pd.DataFrame({"Y": v, "X": v}, index=index)  # noqa: E731
    return MarketData(mk(adv), mk(sigma), mk(half))


def test_cost_components_by_hand():
    f = hand_frame()
    cfg = CostConfig(
        capital=1e8,
        slots=20,
        impact_y=0.5,
        commission_bps=0.5,
        regulatory_bps=0.25,
        borrow_bps_per_year=50.0,
    )
    out = PairCostModel(const_market(f.index), cfg)(f, "Y", "X")
    unit = 1e8 / 20  # $5m per unit of pair capital
    # day 1: trade_y = 1.0 -> Q = $5m, participation 5 % ; trade_x = 0.5 -> Q = $2.5m, participation 2.5 %
    spread = 1.0 * 0.0002 + 0.5 * 0.0002
    impact = 1.0 * 0.5 * 0.02 * np.sqrt(5e6 / 1e8) + 0.5 * 0.5 * 0.02 * np.sqrt(2.5e6 / 1e8)
    commission = 1.5 * 0.75 / 1e4
    assert out["cost_spread"].iloc[1] == pytest.approx(spread)
    assert out["cost_impact"].iloc[1] == pytest.approx(impact)
    assert out["cost_commission"].iloc[1] == pytest.approx(commission)
    assert out["cost_borrow"].iloc[1] == 0.0  # nothing held yet on the entry bar
    assert out["cost"].iloc[1] == pytest.approx(spread + impact + commission)
    # day 2: no trade, but the short X leg (0.5 held) pays borrow
    assert (
        out["cost_spread"].iloc[2]
        == 0
        == out["cost_impact"].iloc[2]
        == out["cost_commission"].iloc[2]
    )
    assert out["cost_borrow"].iloc[2] == pytest.approx(0.5 * 50 / 1e4 / 252)
    assert out["participation"].iloc[1] == pytest.approx(0.05) and unit == 5e6
    assert (out["cost"] >= 0).all() and out["cost"].iloc[0] == 0


def test_a_short_spread_position_borrows_the_y_leg_instead():
    f = hand_frame().assign(w_y=[0, -1.0, -1.0, 0], w_x=[0, 0.5, 0.5, 0])
    out = PairCostModel(const_market(f.index), CostConfig())(f, "Y", "X")
    assert out["cost_borrow"].iloc[2] == pytest.approx(1.0 * 50 / 1e4 / 252)


def test_scaling_laws_and_switches():
    f = hand_frame()
    idx = f.index
    base = PairCostModel(const_market(idx), CostConfig(capital=1e8))(f, "Y", "X")
    big = PairCostModel(const_market(idx), CostConfig(capital=4e8))(f, "Y", "X")
    assert big["cost_impact"].iloc[1] == pytest.approx(2.0 * base["cost_impact"].iloc[1])  # sqrt(4)
    assert big["cost_spread"].iloc[1] == base["cost_spread"].iloc[1]  # linear pieces do not move
    assert big["cost_commission"].iloc[1] == base["cost_commission"].iloc[1]
    doubled_y = PairCostModel(const_market(idx), CostConfig(impact_y=1.0))(f, "Y", "X")
    assert doubled_y["cost_impact"].iloc[1] == pytest.approx(2.0 * base["cost_impact"].iloc[1])
    off = PairCostModel(
        const_market(idx),
        CostConfig(
            impact_y=0, spread_mult=0, commission_bps=0, regulatory_bps=0, borrow_bps_per_year=0
        ),
    )(f, "Y", "X")
    assert (off["cost"] == 0).all()
    liquid = PairCostModel(const_market(idx, adv=4e8), CostConfig())(f, "Y", "X")
    assert liquid["cost_impact"].iloc[1] == pytest.approx(
        0.5 * base["cost_impact"].iloc[1]
    )  # 1/sqrt(4)


def test_costs_are_monotone_in_every_parameter():
    f = hand_frame()

    def mk(**kw):
        return PairCostModel(const_market(f.index), CostConfig(**kw))(f, "Y", "X")["cost"].sum()

    base = mk()
    for kw in (
        dict(impact_y=1.0),
        dict(spread_mult=2.0),
        dict(commission_bps=1.0),
        dict(borrow_bps_per_year=150.0),
        dict(capital=2e8),
    ):
        assert mk(**kw) > base


def test_participation_above_the_cap_is_flagged_not_hidden():
    f = hand_frame()
    out = PairCostModel(const_market(f.index, adv=2e7), CostConfig(max_participation=0.10))(
        f, "Y", "X"
    )
    assert bool(out["breach"].iloc[1]) and out["participation"].iloc[1] == pytest.approx(0.25)
    assert not out["breach"].iloc[2]


def test_a_trade_with_undefined_market_data_is_an_error():
    f = hand_frame()
    mkt = const_market(f.index)
    mkt.adv.iloc[1] = np.nan
    with pytest.raises(ValueError, match="undefined"):
        PairCostModel(mkt, CostConfig())(f, "Y", "X")


def test_breakeven_multiple():
    assert breakeven_multiple(2.0, 0.5) == 4.0 and breakeven_multiple(-1.0, 0.5) == 0.0
    assert breakeven_multiple(1.0, 0.0) == float("inf") and breakeven_multiple(0.0, 0.5) == 0.0


# ---- integration with the walk-forward engine --------------------------------------------------------
SPLIT_CFG = SplitConfig(**SPLIT)
STRAT = StrategyConfig()


@pytest.fixture(scope="module")
def fold(tmp_path_factory, calendar):
    pipe = build(tmp_path_factory.mktemp("costs"), calendar)
    fd = prepare_fold(pipe, make_folds(SPLIT_CFG, calendar, 2013, 2013, 2)[0], SCREEN)
    yield fd
    pipe.close()


def model(fd, **kw):
    cfg = CostConfig(slots=STRAT.slots, **kw)
    return PairCostModel(
        build_market_data(fd.dollar_volume, fd.ret, fd.high, fd.low, fd.close, cfg), cfg
    )


def test_net_is_gross_minus_components_and_aggregation_divides_by_slots(fold):
    pairs = list(zip(fold.screen.selected["y"], fold.screen.selected["x"], strict=True))
    runs = run_pairs(fold, STRAT, pairs, model(fold))
    idx = runs[0].frame.index
    for r in runs:
        f = r.frame
        comp = f[["cost_spread", "cost_commission", "cost_impact", "cost_borrow"]].sum(axis=1)
        assert f["cost"].to_numpy() == pytest.approx(comp.to_numpy()) and (f["cost"] >= 0).all()
        assert (f["net"] - (f["pnl"] - f["cost"])).abs().max() < 1e-15
    agg = aggregate(runs, idx, STRAT.slots)
    assert agg["cost"].sum() == pytest.approx(
        sum(r.frame["cost"].sum() for r in runs) / STRAT.slots
    )
    assert {"cost_spread", "cost_commission", "cost_impact", "cost_borrow"} <= set(agg.columns)
    res = walk_forward([fold], STRAT, cost_fn=model(fold))
    assert res.daily["net"].sum() < res.daily["pnl"].sum() and res.daily["cost"].sum() > 0


def test_gross_is_untouched_by_the_cost_model_and_costs_grow_with_capital(fold):
    gross = walk_forward([fold], STRAT).daily
    lo = walk_forward([fold], STRAT, cost_fn=model(fold, capital=1e7)).daily
    hi = walk_forward([fold], STRAT, cost_fn=model(fold, capital=1e9)).daily
    pd.testing.assert_series_equal(gross["pnl"], lo["pnl"])
    assert hi["cost_impact"].sum() > 5 * lo["cost_impact"].sum()  # ~ sqrt(100) = 10x
    assert hi["cost_spread"].sum() == pytest.approx(lo["cost_spread"].sum())


def test_costs_at_day_t_do_not_depend_on_volume_or_prices_after_t(fold):
    from statarb.backtest.walkforward import FoldData

    cut = pd.Timestamp("2013-08-30")
    pairs = list(zip(fold.screen.selected["y"], fold.screen.selected["x"], strict=True))
    base = run_pairs(fold, STRAT, pairs, model(fold))
    rng = np.random.default_rng(1)
    dv, hi, lo = fold.dollar_volume.copy(), fold.high.copy(), fold.low.copy()
    after = dv.index > cut
    dv.loc[after] *= rng.lognormal(0, 1.0, (after.sum(), dv.shape[1]))
    hi.loc[after] *= 1.1
    lo.loc[after] *= 0.9
    noisy = FoldData(
        fold.fold, fold.screen, fold.logp, fold.ret, fold.event, dv, hi, lo, fold.close
    )
    other = run_pairs(noisy, STRAT, pairs, model(noisy))
    for a, b in zip(base, other, strict=True):
        cols = ["cost_spread", "cost_commission", "cost_impact", "cost_borrow", "cost"]
        pd.testing.assert_frame_equal(a.frame.loc[:cut, cols], b.frame.loc[:cut, cols])
    assert any(
        not np.allclose(a.frame.loc[cut:, "cost_impact"], b.frame.loc[cut:, "cost_impact"])
        for a, b in zip(base, other, strict=True)
        if a.frame.loc[cut:, "cost_impact"].sum() > 0
    )


def test_impact_is_linear_in_volatility_and_uses_each_legs_own_inputs():
    f = hand_frame()
    idx = f.index
    base = PairCostModel(const_market(idx, sigma=0.02), CostConfig())(f, "Y", "X")
    vol = PairCostModel(const_market(idx, sigma=0.06), CostConfig())(f, "Y", "X")
    assert vol["cost_impact"].iloc[1] == pytest.approx(3.0 * base["cost_impact"].iloc[1])
    # different inputs per leg: Y volatile and thin, X calm and deep
    mk = const_market(idx)
    mk.sigma["Y"], mk.sigma["X"], mk.adv["Y"], mk.adv["X"] = 0.04, 0.01, 4e7, 4e8
    out = PairCostModel(mk, CostConfig(impact_y=1.0))(f, "Y", "X")
    expect = 1.0 * 1.0 * 0.04 * np.sqrt(5e6 / 4e7) + 0.5 * 1.0 * 0.01 * np.sqrt(2.5e6 / 4e8)
    assert out["cost_impact"].iloc[1] == pytest.approx(expect)
    assert out["participation"].iloc[1] == pytest.approx(
        5e6 / 4e7
    )  # the larger of the two legs' shares


def test_a_cost_factory_builds_one_cost_function_per_fold_and_matches_a_plain_cost_fn(fold):
    built = []

    def factory(fd):
        built.append(fd.fold.index)
        return model(fd)

    a = walk_forward([fold], STRAT, cost_factory=factory).daily
    b = walk_forward([fold], STRAT, cost_fn=model(fold)).daily
    pd.testing.assert_frame_equal(a, b)
    assert built == [fold.fold.index]
    with pytest.raises(ValueError, match="not both"):
        walk_forward([fold], STRAT, cost_fn=model(fold), cost_factory=factory)
