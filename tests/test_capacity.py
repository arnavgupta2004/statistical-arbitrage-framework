"""Capacity from cost components: the analytic scaling must equal a real re-run at another capital."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from statarb.backtest.capacity import (
    break_even_capital,
    net_at_capital,
    participation_breach_share,
)
from statarb.backtest.costs import CostConfig, MarketData, PairCostModel
from statarb.backtest.pca_walkforward import PCAStrategyConfig, run_fold
from statarb.signals.residuals import ResidualConfig

from .test_pca_walkforward import factor_world

CFG = PCAStrategyConfig(
    residual=ResidualConfig(
        n_factors=2, fit_window=250, refit_every=21, beta_window=40, signal="reversal"
    )
)


def realistic(fs, seed=0):
    rng = np.random.default_rng(seed)
    shape = fs.tradable.shape
    market = {
        "adv": rng.uniform(2e7, 4e8, shape),
        "sigma": rng.uniform(0.008, 0.025, shape),
        "half_spread": rng.choice([1e-4, 1.5e-4, 2.5e-4, 5e-4], shape),
    }
    return replace(fs, market=market)


def test_analytic_capacity_reproduces_a_rerun_at_another_capital():
    fs = realistic(factor_world(phi=0.5))
    base = run_fold(fs, CFG, CostConfig(capital=1e8))
    for capital in (1e7, 3e8, 2e9):
        rerun = run_fold(fs, CFG, CostConfig(capital=capital))
        analytic = net_at_capital(base, capital, 1e8)
        assert np.allclose(analytic, rerun["net"], atol=1e-13, rtol=1e-10)
        assert participation_breach_share(
            base["participation"], capital, 1e8, 0.1
        ) == pytest.approx(float((rerun["participation"] > 0.1).mean()))
    assert base["participation"].max() > 0.02  # the check is not vacuous


def test_the_same_scaling_holds_for_the_pair_cost_model():
    idx = pd.bdate_range("2020-01-01", periods=8)
    rng = np.random.default_rng(1)
    adv = pd.DataFrame({"Y": [3e7] * 8, "X": [8e8] * 8}, index=idx)
    sig = pd.DataFrame(rng.uniform(0.01, 0.02, (8, 2)), index=idx, columns=["Y", "X"])
    hs = pd.DataFrame({"Y": [2.5e-4] * 8, "X": [1e-4] * 8}, index=idx)
    frame = pd.DataFrame(
        {
            "trade_y": [1.0, 0, 0.1, 0, 0, 0, 0, 1.0],
            "trade_x": [0.7, 0, 0.07, 0, 0, 0, 0, 0.7],
            "w_y": [1.0] * 7 + [0.0],
            "w_x": [-0.7] * 7 + [0.0],
        },
        index=idx,
    )
    frame["trade"] = frame["trade_y"] + frame["trade_x"]
    lo = PairCostModel(MarketData(adv, sig, hs), CostConfig(capital=1e8))(frame, "Y", "X")
    hi = PairCostModel(MarketData(adv, sig, hs), CostConfig(capital=4e8))(frame, "Y", "X")
    assert hi["cost_impact"].sum() == pytest.approx(2 * lo["cost_impact"].sum())  # sqrt(4)
    for c in ("cost_spread", "cost_commission", "cost_borrow"):
        assert hi[c].sum() == pytest.approx(lo[c].sum())
    assert (hi["participation"] / lo["participation"]).dropna().round(9).unique().tolist() == [4.0]


def frame(gross, spread, comm, impact, borrow, n=50):
    return pd.DataFrame(
        {
            "pnl": [gross] * n,
            "cost_spread": [spread] * n,
            "cost_commission": [comm] * n,
            "cost_impact": [impact] * n,
            "cost_borrow": [borrow] * n,
        }
    )


def test_break_even_capital_and_its_edge_cases():
    d = frame(
        10e-4, 2e-4, 1e-4, 3e-4, 1e-4
    )  # room = 6e-4 against 3e-4 of impact at 1e8 -> 4x capital
    c = break_even_capital(d, 1e8)
    assert c == pytest.approx(4e8)
    assert net_at_capital(d, c, 1e8).mean() == pytest.approx(0.0, abs=1e-15)
    assert net_at_capital(d, c / 2, 1e8).mean() > 0 > net_at_capital(d, c * 2, 1e8).mean()
    assert (
        break_even_capital(frame(3e-4, 2e-4, 1e-4, 3e-4, 1e-4), 1e8) == 0.0
    )  # fixed costs eat the gross
    assert break_even_capital(frame(10e-4, 2e-4, 1e-4, 0.0, 1e-4), 1e8) == float("inf")
    # cheaper spreads and impact move it out; the multipliers act on their own component only
    assert break_even_capital(d, 1e8, spread_mult=0.0) == pytest.approx(1e8 * (8e-4 / 3e-4) ** 2)
    assert break_even_capital(d, 1e8, impact_mult=0.5) == pytest.approx(1e8 * (6e-4 / 1.5e-4) ** 2)
    assert break_even_capital(d, 1e8, commission_mult=0.0, borrow_mult=0.0) == pytest.approx(
        1e8 * (8e-4 / 3e-4) ** 2
    )
    assert net_at_capital(d, 1e8, 1e8, spread_mult=2.0).mean() == pytest.approx(
        d["pnl"].mean() - 9e-4  # 2 x 2 + 1 + 3 + 1
    )
    assert net_at_capital(d, 4e8, 1e8, impact_mult=0.0).mean() == pytest.approx(
        d["pnl"].mean() - 4e-4
    )


def test_break_even_when_the_fixed_costs_exactly_equal_the_gross():
    assert break_even_capital(frame(4e-4, 2e-4, 1e-4, 0.0, 1e-4), 1e8) == 0.0
    assert break_even_capital(frame(4e-4, 2e-4, 1e-4, 3e-4, 1e-4), 1e8) == 0.0
