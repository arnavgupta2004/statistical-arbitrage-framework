"""P&L accounting, sizing and the end-to-end causality of the whole strategy chain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.backtest.pair_pnl import freeze_at_entry, pair_returns
from statarb.data.leakage import assert_no_lookahead
from statarb.models.hedge_ratio import HedgeSpec, hedge_path
from statarb.portfolio.construction import entry_notional, leg_weights, spread_returns
from statarb.signals.pairs import PairParams, generate_positions, trades_table
from statarb.signals.zscore import spread_zscore

from .test_hedge_ratio import pair


def series(values, idx=None):
    idx = idx if idx is not None else pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def test_pnl_by_hand_with_the_one_bar_lag():
    ry = series([0, 0.01, 0.02, -0.01, 0.03])
    rx = series([0, 0.00, 0.01, 0.01, 0.01])
    beta = series([0.5] * 5)
    pos = np.array([0, 1, 1, 0, 0])  # long the spread from the close of bar 1 to the close of bar 3
    out = pair_returns(pos, beta, np.where(pos != 0, 1.0, 0.0), ry, rx)
    # bars 2 and 3 are earned (position held over them); bar 1 (the entry bar) is NOT
    expected = [0, 0, 0.02 - 0.5 * 0.01, -0.01 - 0.5 * 0.01, 0]
    assert out["pnl"].to_numpy() == pytest.approx(expected)
    assert out["w_y"].tolist() == [0, 1, 1, 0, 0] and out["w_x"].tolist() == [0, -0.5, -0.5, 0, 0]


def test_a_return_on_the_entry_bar_is_never_earned():
    ry, rx = series([0, 0.5, 0, 0]), series([0, 0.0, 0, 0])
    out = pair_returns(
        np.array([0, 1, 1, 0]), series([1.0] * 4), np.array([0, 1.0, 1.0, 0]), ry, rx
    )
    assert out["pnl"].sum() == 0.0  # the +50 % happened on the bar the position was opened


def test_turnover_counts_entry_exit_and_drift_rebalancing():
    ry, rx = series([0, 0, 0.10, 0]), series([0, 0, 0.0, 0])
    beta = series([0.5] * 4)
    out = pair_returns(np.array([0, 1, 1, 0]), beta, np.array([0, 1.0, 1.0, 0]), ry, rx)
    t = out["trade"].to_numpy()
    assert t[1] == pytest.approx(1.5)  # entry: |w_y| + |w_x| = 1 + 0.5
    assert t[2] == pytest.approx(0.10)  # w_y drifted 1 -> 1.10, traded back to 1.0 (w_x unchanged)
    assert t[3] == pytest.approx(1.5)  # exit: 1 + 0.5 (both legs closed at target weights)
    flat = pair_returns(
        np.array([0, 1, 1, 0]), beta, np.array([0, 1.0, 1.0, 0]), series([0] * 4), series([0] * 4)
    )
    assert flat["trade"].tolist() == pytest.approx(
        [0, 1.5, 0, 1.5]
    )  # no drift, no hedge change: nothing else


def test_a_changing_hedge_ratio_trades_the_x_leg_and_freezing_removes_it():
    z = series([0] * 5)
    beta = series([0.5, 0.5, 0.6, 0.7, 0.7])
    pos, n = np.array([0, 1, 1, 1, 0]), np.array([0, 1.0, 1.0, 1.0, 0])
    live = pair_returns(pos, beta, n, z, z)
    frozen = pair_returns(pos, beta, n, z, z, freeze_beta=True)
    assert live["trade"].tolist() == pytest.approx([0, 1.5, 0.1, 0.1, 1.7])
    assert frozen["trade"].tolist() == pytest.approx([0, 1.5, 0, 0, 1.5])
    assert frozen["w_x"].tolist() == [0, -0.5, -0.5, -0.5, 0]
    assert freeze_at_entry(np.array([0.5, 0.5, 0.9, 0.9]), np.array([0, 1, 1, 0])).tolist() == [
        0,
        0.5,
        0.5,
        0,
    ]


def test_pnl_conserves_across_trades_and_matches_the_trades_table():
    ly, lx, _ = pair(900, seed=3)
    ry, rx = ly.diff().fillna(0), lx.diff().fillna(0)
    beta = hedge_path(ly, lx, HedgeSpec("static"), ly.index[499])["beta"]
    z = spread_zscore(ly, lx, beta, 60)["z"]
    res = generate_positions(z.to_numpy(), PairParams(), 500)
    out = pair_returns(res.position, beta, np.abs(res.position).astype(float), ry, rx)
    tr = trades_table(ly.index, res, out["pnl"])
    assert len(tr) > 3 and tr["pnl"].sum() == pytest.approx(out["pnl"].sum())
    assert (out["pnl"].iloc[:500] == 0).all()  # nothing earned before trade_start


def test_leg_weights_are_dollar_hedged_and_vol_sizing_is_frozen_within_a_trade():
    w_y, w_x = leg_weights([1, -1, 0], [0.8, 0.8, 0.8], [2.0, 2.0, 0.0])
    assert w_y.tolist() == [2.0, -2.0, 0.0] and w_x.tolist() == pytest.approx([-1.6, 1.6, 0.0])
    rng = np.random.default_rng(0)
    sr = pd.Series(
        rng.normal(0, np.linspace(0.005, 0.02, 200)),
        index=pd.bdate_range("2020-01-01", periods=200),
    )
    pos = np.zeros(200, dtype=int)
    pos[80:120], pos[150:170] = 1, -1
    n = entry_notional(pos, sr, target_vol=0.01, window=60, max_notional=3.0)
    assert (
        (n[80:120] == n[80]).all() and (n[150:170] == n[150]).all() and n[80] != n[150]
    )  # frozen per trade
    assert n[80] == pytest.approx(0.01 / sr.iloc[21:81].std())  # window ending at the ENTRY bar
    assert (n[pos == 0] == 0).all()
    assert (
        entry_notional(pos, sr, target_vol=10.0, max_notional=3.0)[80] == 3.0
    )  # capped, not 1000x
    assert entry_notional(pos, sr)[80] == 1.0  # unit sizing


def test_spread_returns_use_the_hedge_ratio_held_overnight():
    beta = series([0.5, 0.6, 0.7])
    out = spread_returns(series([0, 0.02, 0.03]), series([0, 0.01, 0.01]), beta)
    assert np.isnan(out.iloc[0]) and out.iloc[1] == pytest.approx(0.02 - 0.5 * 0.01)
    assert out.iloc[2] == pytest.approx(0.03 - 0.6 * 0.01)


def test_undefined_inputs_are_errors_not_silent_zeros():
    r = series([0, 0.01, 0.01])
    with pytest.raises(ValueError, match="hedge ratio is undefined"):
        pair_returns(np.array([0, 1, 1]), series([np.nan, np.nan, 0.5]), np.ones(3), r, r)
    with pytest.raises(ValueError, match="return is missing"):
        pair_returns(
            np.array([0, 1, 1]), series([0.5] * 3), np.ones(3), series([0, 0.01, np.nan]), r
        )


# ---- the whole chain is causal, for every hedge method ---------------------------------------
CHAIN = [
    HedgeSpec("static"),
    HedgeSpec("expanding"),
    HedgeSpec("rolling", window=60),
    HedgeSpec("rolling", window=250),
    HedgeSpec("kalman", delta=1e-5),
]


@pytest.mark.parametrize("spec", CHAIN, ids=lambda s: s.label)
def test_the_full_strategy_chain_is_causal(spec):
    """prices -> hedge ratio -> z-score -> positions -> P&L: scramble or drop the future, nothing earlier moves."""
    ly, lx, _ = pair(800, seed=9, drift_sd=0.002)
    frame = pd.DataFrame({"ly": ly, "lx": lx})
    train_end = frame.index[399]

    def chain(f: pd.DataFrame) -> pd.DataFrame:
        beta = hedge_path(f["ly"], f["lx"], spec, train_end)["beta"]
        z = spread_zscore(f["ly"], f["lx"], beta, 60)["z"]
        res = generate_positions(z.to_numpy(), PairParams(), 400)
        ry, rx = f["ly"].diff().fillna(0.0), f["lx"].diff().fillna(0.0)
        out = pair_returns(res.position, beta, np.abs(res.position).astype(float), ry, rx)
        return out[["pnl", "trade", "w_y", "w_x"]]

    assert_no_lookahead(chain, frame, frame.index[[450, 550, 650, 750]], atol=1e-10)
