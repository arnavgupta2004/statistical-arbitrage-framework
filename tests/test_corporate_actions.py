"""Corporate-action identities, checked against hand-computed and analytically known cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.config import ValidationConfig
from statarb.data.cleaning.alignment import build_panel
from statarb.data.cleaning.corporate_actions import (
    as_traded_close,
    assign_dividends_to_rows,
    dividend_growth,
    future_split_factor,
    rebase_to_as_of,
)
from statarb.data.cleaning.validation import reconcile_vendor_adjustment
from statarb.data.sources.base import empty_actions
from statarb.data.sources.synthetic import SyntheticSource

from .conftest import DIVIDEND_DATES, make_world, ts


def actions(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "kind", "value"]).assign(
        date=lambda d: d["date"].astype("datetime64[ns]")
    )


def test_as_traded_close_matches_the_aapl_split():
    # AAPL 4-for-1 on 2020-08-31; Yahoo's pre-split close of 124.8250 is the as-traded 499.30 / 4.
    idx = pd.DatetimeIndex(["2020-08-27", "2020-08-28", "2020-08-31", "2020-09-01"])
    close = pd.Series([125.010002, 124.807503, 129.039993, 134.179993], index=idx)
    raw = as_traded_close(close, actions(("2020-08-31", "split", 4.0)))
    assert raw.iloc[0] == pytest.approx(125.010002 * 4)
    assert raw.iloc[1] == pytest.approx(124.807503 * 4)
    assert raw.iloc[2] == pytest.approx(129.039993)  # the ex-date itself is already post-split
    assert raw.iloc[3] == pytest.approx(134.179993)


def test_as_traded_close_compounds_multiple_and_reverse_splits():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.Series(10.0, index=idx)
    acts = actions(
        ("2020-01-03", "split", 2.0), ("2020-01-05", "split", 0.1)
    )  # 2:1 then 1:10 reverse
    raw = as_traded_close(close, acts)
    assert raw.tolist() == pytest.approx([10 * 2 * 0.1, 10 * 2 * 0.1, 10 * 0.1, 10 * 0.1, 10, 10])


def test_future_split_factor_only_counts_strictly_later_splits():
    a = actions(("2020-01-03", "split", 2.0), ("2020-01-05", "split", 3.0))
    assert future_split_factor(a, ts("2020-01-02")) == 6.0
    assert (
        future_split_factor(a, ts("2020-01-03")) == 3.0
    )  # a split on the as_of day is already applied
    assert future_split_factor(a, ts("2020-01-05")) == 1.0
    assert future_split_factor(empty_actions(), ts("2020-01-05")) == 1.0


def test_total_return_with_a_dividend_by_hand():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
    prices = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": [100.0, 101.0, 99.0], "volume": 10.0},
        index=idx,
    )
    panel = build_panel({"X": (prices, actions(("2020-01-06", "dividend", 2.0)))}, idx)
    assert panel.ret["X"].iloc[1] == pytest.approx(101 / 100 - 1)
    assert panel.ret["X"].iloc[2] == pytest.approx(
        (99 + 2) / 101 - 1
    )  # dividend counted in the return
    assert np.isnan(panel.ret["X"].iloc[0])


def test_dividend_growth_reproduces_total_return():
    """I_t = C_t * G_t satisfies I_t / I_{t-1} = 1 + r_t exactly."""
    idx = pd.date_range("2020-01-01", periods=8, freq="B")
    close = pd.Series([50, 51, 50.5, 52, 51.5, 53, 52, 54.0], index=idx)
    div = pd.Series(0.0, index=idx)
    div.iloc[3], div.iloc[6] = 0.8, 1.1
    growth = dividend_growth(close, div)
    index_level = close * growth
    ret = (close + div) / close.shift(1) - 1
    np.testing.assert_allclose(
        (index_level / index_level.shift(1) - 1).iloc[1:], ret.iloc[1:], rtol=1e-12
    )


def test_assign_dividends_to_next_available_bar_and_drops_future_ones():
    rows = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-07"])  # 01-06 bar missing
    divs = actions(("2020-01-06", "dividend", 1.0), ("2020-01-10", "dividend", 5.0))
    out = assign_dividends_to_rows(rows, divs)
    assert out.tolist() == [
        0.0,
        0.0,
        1.0,
    ]  # missing ex-date bar -> paid on next bar; post-sample dividend dropped


def test_rebasing_a_later_download_reproduces_the_earlier_download_exactly():
    """The core point-in-time identity: rebase(vendor at V2, as_of=V1) == vendor at V1."""
    world = make_world()
    v1, v2 = ts("2016-02-26"), ts("2016-12-30")  # split (2016-03-15) falls between
    early = SyntheticSource(world, vendor_date=v1).fetch("AAA", ts("2015-01-01"), v2)
    late = SyntheticSource(world, vendor_date=v2).fetch("AAA", ts("2015-01-01"), v2)
    assert late.prices["close"].iloc[0] != pytest.approx(
        early.prices["close"].iloc[0]
    )  # bases differ

    p, a = rebase_to_as_of(late.prices.drop(columns="adj_close"), late.actions, v1)
    pd.testing.assert_frame_equal(p, early.prices.drop(columns="adj_close"), check_freq=False)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), early.actions.reset_index(drop=True))


def test_returns_are_invariant_to_the_vendor_basis_but_levels_are_not():
    world = make_world()
    early = SyntheticSource(world, vendor_date="2016-02-26").fetch(
        "AAA", ts("2015-01-01"), ts("2016-02-26")
    )
    late = SyntheticSource(world, vendor_date="2016-12-30").fetch(
        "AAA", ts("2015-01-01"), ts("2016-02-26")
    )
    r_early = early.prices["close"].pct_change().dropna()
    r_late = late.prices["close"].pct_change().dropna()
    np.testing.assert_allclose(r_early, r_late, rtol=1e-10)
    ratio = (late.prices["close"] / early.prices["close"]).to_numpy()
    # a *constant* factor: the later download has already divided pre-split prices by the 2-for-1
    np.testing.assert_allclose(ratio, 0.5)


def test_dividend_before_split_is_rescaled_consistently():
    world = make_world()
    late = SyntheticSource(world, vendor_date="2016-12-30").fetch(
        "AAA", ts("2015-01-01"), ts("2016-12-30")
    )
    divs = late.actions[late.actions["kind"] == "dividend"].set_index("date")["value"]
    assert divs[ts(DIVIDEND_DATES[0])] == pytest.approx(0.5)  # E-basis amount, vendor at world end
    early = SyntheticSource(world, vendor_date="2016-02-26").fetch(
        "AAA", ts("2015-01-01"), ts("2016-02-26")
    )
    d_early = early.actions[early.actions["kind"] == "dividend"].set_index("date")["value"]
    assert d_early[ts(DIVIDEND_DATES[0])] == pytest.approx(0.5 * 2.0)  # per pre-split share


def test_reconcile_with_vendor_adj_close_is_clean_on_correct_data():
    world = make_world()
    raw = SyntheticSource(world).fetch("BBB", ts("2015-01-01"), ts("2016-12-30"))
    assert reconcile_vendor_adjustment("BBB", raw.prices, raw.actions, ValidationConfig()) == []


def test_reconcile_flags_a_corrupted_vendor_adjustment():
    raw = SyntheticSource(make_world()).fetch("BBB", ts("2015-01-01"), ts("2016-12-30"))
    bad = raw.prices.copy()
    bad.iloc[300:, bad.columns.get_loc("adj_close")] *= (
        1.05  # vendor silently rescales part of history
    )
    issues = reconcile_vendor_adjustment("BBB", bad, raw.actions, ValidationConfig())
    assert [i.code for i in issues] == ["ADJ_CLOSE_MISMATCH"]


def test_tr_close_is_anchored_and_uses_no_future_dividend(make_pipeline):
    pipe = make_pipeline()
    pipe.refresh()
    panel = pipe.panel(["BBB"], "2015-01-02", "2016-12-30")
    anchor = ts("2015-12-31")
    tr = panel.tr_close(anchor)["BBB"]
    assert tr.index[-1] == anchor
    assert tr.iloc[-1] == pytest.approx(panel.close["BBB"].loc[anchor])  # anchored to the close
    # anchored series must not change if all later dividends are altered
    tampered = pipe.panel(["BBB"], "2015-01-02", "2016-12-30")
    tampered.growth.loc["2016-01-14":] *= 3.0  # what a later dividend would do
    pd.testing.assert_series_equal(tampered.tr_close(anchor)["BBB"], tr)
    # and consecutive TR-price ratios equal 1 + total return
    ratio = (tr / tr.shift(1) - 1).iloc[1:]
    np.testing.assert_allclose(ratio, panel.ret["BBB"].loc[tr.index].iloc[1:], atol=1e-12)
