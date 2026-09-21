"""Trading state machine: hand-worked sequences, invariants (property test) and causality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from statarb.data.leakage import assert_no_lookahead
from statarb.signals.pairs import PairParams, generate_positions, trades_table

P = PairParams(entry=2.0, exit=0.5, stop=4.0, max_hold=60)


def run(z, params=P, start=0):
    return generate_positions(np.array(z, dtype=float), params, start)


def test_long_spread_trade_by_hand():
    #        0    1     2     3     4     5    6
    z = [0.0, -1.0, -2.5, -1.5, -0.7, -0.3, 0.0]
    r = run(z)
    assert r.position.tolist() == [0, 0, 1, 1, 1, 0, 0]  # enter at z=-2.5, exit once z >= -0.5
    assert r.exit_reason.tolist() == [0, 0, 0, 0, 0, 1, 0]  # closed on bar 5, reason "mean"
    assert r.entry_z[2] == -2.5 and np.isnan(r.entry_z[3])


def test_short_spread_trade_is_the_mirror_image():
    r = run([0.0, 1.0, 2.5, 1.5, 0.7, 0.3, 0.0])
    assert r.position.tolist() == [0, 0, -1, -1, -1, 0, 0] and r.exit_reason[5] == 1
    mirror = run([-v for v in [0.0, 1.0, 2.5, 1.5, 0.7, 0.3, 0.0]])
    assert (mirror.position == -r.position).all()


def test_thresholds_are_strict_inequalities():
    assert run([0, 2.0, 0]).position.sum() == 0  # |z| == entry does not enter
    assert run([0, 2.0001, 2.0, 0.5000001, 0.5]).position.tolist() == [
        0,
        -1,
        -1,
        -1,
        0,
    ]  # exit at <= exit


def test_stop_loss_exits_and_blocks_immediate_reentry_until_the_spread_neutralises():
    z = [0, -2.5, -3.0, -4.2, -2.5, -2.6, 0.2, -2.5, -0.2]
    r = run(z)
    assert r.position.tolist() == [0, 1, 1, 0, 0, 0, 0, 1, 0]
    assert r.exit_reason[3] == 2  # stop
    # bars 4-5: still stretched in the same direction -> blocked, no re-entry; bar 6 (0.2) unblocks
    assert r.position[4] == 0 and r.position[5] == 0 and r.position[7] == 1
    assert r.exit_reason[8] == 1


def test_a_stop_in_one_direction_does_not_block_the_opposite_direction():
    r = run([0, -2.5, -4.5, 2.5, 0.0])
    assert (
        r.exit_reason[2] == 2 and r.position[3] == -1
    )  # long stopped; short entered on the next bar


def test_time_stop_exits_after_max_hold_bars_and_blocks_reentry():
    p = PairParams(2.0, 0.5, 4.0, max_hold=3)
    r = run([0, -2.5, -2.4, -2.3, -2.2, -2.1, 0.0, -2.5], p)
    assert r.position.tolist() == [0, 1, 1, 1, 0, 0, 0, 1]  # held bars 1,2,3 -> exits on bar 4
    assert r.exit_reason[4] == 3
    assert r.position[5] == 0  # still stretched (z=-2.1): blocked until it neutralises


def test_no_entry_beyond_the_stop_level():
    assert run([0, -4.5, -4.5, -2.5, -0.1]).position.tolist() == [0, 0, 0, 1, 0]


def test_nothing_is_traded_before_trade_start():
    z = [-2.5, -2.5, -2.5, -2.5, -0.1]
    r = run(z, start=2)
    assert r.position.tolist() == [0, 0, 1, 1, 0]  # the same z-scores earlier would have entered
    assert (run(z, start=5).position == 0).all()


def test_missing_z_flattens_a_position_and_stays_flat_while_missing():
    z = [0, -2.5, -2.0, np.nan, np.nan, -2.5, -0.1]
    r = run(z)
    assert r.position.tolist() == [0, 1, 1, 0, 0, 1, 0]
    assert r.exit_reason[3] == 4  # "data"


def test_no_same_bar_reversal():
    r = run([0, -2.5, 2.5, 2.5, 0])
    # bar 2: the long exits (u = -2.5 <= exit) and no short opens on the same bar; it enters on bar 3
    assert r.position.tolist() == [0, 1, 0, -1, 0] and r.exit_reason[2] == 1


def test_pair_params_validation():
    for bad in (
        dict(entry=1.0, exit=1.5),
        dict(entry=3.0, stop=2.0),
        dict(exit=-0.1),
        dict(max_hold=0),
    ):
        with pytest.raises(ValueError):
            PairParams(**{**dict(entry=2.0, exit=0.5, stop=4.0, max_hold=10), **bad})
    PairParams(entry=2.0, exit=0.0, stop=4.0)  # exit at the mean is allowed


@settings(max_examples=150, deadline=None)
@given(
    st.lists(st.floats(-6, 6, allow_nan=False), min_size=1, max_size=200),
    st.floats(0.5, 3.0),
    st.integers(1, 30),
    st.integers(0, 50),
)
def test_state_machine_invariants(z, entry, max_hold, start):
    p = PairParams(entry=entry, exit=entry / 4, stop=entry * 2 + 0.1, max_hold=max_hold)
    r = generate_positions(np.array(z), p, start)
    pos, zz = r.position, np.array(z)
    assert set(np.unique(pos)) <= {-1, 0, 1} and (pos[:start] == 0).all()
    entries = np.flatnonzero((pos != 0) & (np.r_[0, pos[:-1]] == 0))
    for i in entries:
        assert p.entry < abs(zz[i]) < p.stop and pos[i] == (1 if zz[i] < 0 else -1)
        assert r.entry_z[i] == zz[i]
    # a trade never lasts more than max_hold bars
    run_len = 0
    for i in range(len(pos)):
        run_len = (
            run_len + 1
            if pos[i] != 0 and (i == 0 or pos[i - 1] == pos[i])
            else (1 if pos[i] != 0 else 0)
        )
        assert run_len <= p.max_hold
    # every exit is explained by exactly one reason, and only positions can exit
    exits = np.flatnonzero((pos == 0) & (np.r_[0, pos[:-1]] != 0))
    assert (r.exit_reason[exits] != 0).all() and (r.exit_reason != 0).sum() == len(exits)
    # after a stop/time exit the same direction cannot re-enter before |z| <= exit has occurred
    for e in exits:
        if r.exit_reason[e] in (2, 3):
            d = pos[e - 1]
            for j in range(e + 1, len(pos)):
                if abs(zz[j]) <= p.exit:
                    break
                assert not (pos[j] == d and (pos[j - 1] == 0))


def test_positions_are_causal():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    z = pd.DataFrame({"z": np.cumsum(rng.normal(0, 0.4, 300)) * 0.3}, index=idx)

    def fn(f):
        return pd.DataFrame(
            {"pos": generate_positions(f["z"].to_numpy(), P, 20).position}, index=f.index
        )

    assert_no_lookahead(fn, z, idx[[60, 120, 200, 280]])


def test_trades_table_by_hand_including_pnl_and_an_open_trade():
    idx = pd.bdate_range("2020-01-01", periods=12)
    z = [0, -2.5, -1, -0.2, 0, 2.5, 3.0, 4.5, 0, 0, -2.5, -2.4]
    r = generate_positions(np.array(z), P)
    pnl = pd.Series(np.arange(12, dtype=float) * 0.01, index=idx)
    t = trades_table(idx, r, pnl)
    assert t["direction"].tolist() == [1, -1, 1]
    assert t["exit_reason"].tolist() == ["mean", "stop", "open"]
    assert t["entry"].tolist() == [idx[1], idx[5], idx[10]]
    assert (
        t["exit"].iloc[0] == idx[3] and t["exit"].iloc[1] == idx[7] and pd.isna(t["exit"].iloc[2])
    )
    assert t["bars_held"].tolist() == [2, 2, 1]
    # trade 1 holds after the close of bar 1 and earns bars 2..3 (exit bar included)
    assert t["pnl"].iloc[0] == pytest.approx(pnl.iloc[2:4].sum()) and t["pnl"].iloc[
        2
    ] == pytest.approx(pnl.iloc[11])
    assert trades_table(idx, generate_positions(np.zeros(12), P)).empty


def test_a_block_survives_a_forced_exit_in_the_opposite_direction():
    """Regression (found by the property test): the short blocked by its time stop must stay blocked
    when a long is entered and time-stopped in between, until |z| <= exit has actually occurred."""
    p = PairParams(entry=2.0, exit=0.5, stop=4.1, max_hold=1)
    z = [
        3.0,
        1.0,
        -3.0,
        -1.0,
        3.0,
    ]  # short, time exit | long, time exit | short again: still blocked
    r = generate_positions(np.array(z), p, 0)
    assert r.position.tolist() == [-1, 0, 1, 0, 0]
    assert r.exit_reason.tolist() == [0, 3, 0, 3, 0]
    # once |z| <= exit has been seen both directions are free again
    r2 = generate_positions(np.array(z[:4] + [0.2, 3.0]), p, 0)
    assert r2.position.tolist() == [-1, 0, 1, 0, 0, -1]
    # a stop block on the short and a time block on the long are kept independently
    z3 = np.array([3.0, 5.0, -3.0, -1.0, -2.5, 3.0, 0.2, 3.0])
    r3 = generate_positions(z3, PairParams(2.0, 0.5, 4.1, 1), 0)
    assert r3.position.tolist() == [-1, 0, 1, 0, 0, 0, 0, -1]
    assert r3.exit_reason.tolist() == [0, 2, 0, 3, 0, 0, 0, 0]
