from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.data.cleaning.alignment import build_panel
from statarb.data.sources.base import empty_actions

from .conftest import ts


def bars(dates, close=100.0, volume=1000.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates, name="date")
    c = np.full(len(idx), close) if np.isscalar(close) else np.asarray(close, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c, "low": c, "close": c, "volume": np.full(len(idx), volume)}, index=idx
    )


SESSIONS = pd.bdate_range("2020-01-06", periods=10)  # a plain 10-day calendar


def test_missing_bars_stay_nan_and_are_never_forward_filled():
    present = SESSIONS.delete([3, 4])
    panel = build_panel({"X": (bars(present, np.arange(1, 9) + 100.0), empty_actions())}, SESSIONS)
    assert panel.close["X"].isna().tolist() == [
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert panel.observed["X"].tolist() == [not v for v in panel.close["X"].isna()]


def test_return_is_defined_only_between_consecutive_sessions():
    present = SESSIONS.delete([3])
    panel = build_panel({"X": (bars(present, np.linspace(100, 108, 9)), empty_actions())}, SESSIONS)
    r = panel.ret["X"]
    assert np.isnan(r.iloc[3])  # no bar
    assert np.isnan(
        r.iloc[4]
    )  # bar exists but yesterday's is missing: a 2-day move, not a daily return
    assert r.iloc[5] == pytest.approx(panel.close["X"].iloc[5] / panel.close["X"].iloc[4] - 1)


def test_listing_and_delisting_produce_nan_outside_the_lifespan():
    late = build_panel(
        {"L": (bars(SESSIONS[4:]), empty_actions()), "D": (bars(SESSIONS[:6]), empty_actions())},
        SESSIONS,
    )
    assert late.close["L"].iloc[:4].isna().all() and late.close["L"].iloc[4:].notna().all()
    assert late.close["D"].iloc[6:].isna().all() and late.close["D"].iloc[:6].notna().all()


def test_eligibility_excludes_late_listings_and_early_delistings():
    p = build_panel(
        {
            "FULL": (bars(SESSIONS), empty_actions()),
            "LATE": (bars(SESSIONS[5:]), empty_actions()),
            "GONE": (bars(SESSIONS[:5]), empty_actions()),
            "HOLEY": (bars(SESSIONS.delete([2])), empty_actions()),
        },
        SESSIONS,
    )
    assert p.eligible(SESSIONS[0], SESSIONS[-1], min_coverage=1.0) == ["FULL"]
    assert p.eligible(SESSIONS[0], SESSIONS[-1], min_coverage=0.9) == ["FULL", "HOLEY"]
    assert p.eligible(SESSIONS[5], SESSIONS[-1], min_coverage=1.0) == ["FULL", "HOLEY", "LATE"]


def test_bars_off_the_requested_calendar_are_ignored():
    weekend = pd.DatetimeIndex(list(SESSIONS[:3]) + [ts("2020-01-11")])
    p = build_panel({"X": (bars(weekend), empty_actions())}, SESSIONS)
    assert p.observed["X"].sum() == 3 and ts("2020-01-11") not in p.close.index


def test_dividend_is_credited_to_the_next_bar_when_the_ex_date_bar_is_missing():
    present = SESSIONS.delete([4])
    acts = pd.DataFrame({"date": [SESSIONS[4]], "kind": ["dividend"], "value": [1.5]}).astype(
        {"date": "datetime64[ns]"}
    )
    p = build_panel({"X": (bars(present), acts)}, SESSIONS)
    assert p.dividends["X"].iloc[5] == 1.5 and p.dividends["X"].sum() == 1.5


def test_dollar_volume_and_window_and_coverage():
    p = build_panel({"X": (bars(SESSIONS, 10.0, 500.0), empty_actions())}, SESSIONS)
    assert (p.dollar_volume["X"] == 5000.0).all()
    w = p.window(SESSIONS[2], SESSIONS[5])
    assert len(w.close) == 4 and w.coverage()["X"] == 1.0
    assert p.coverage(SESSIONS[0], SESSIONS[4])["X"] == 1.0


def test_tickers_with_no_rows_stay_as_all_nan_columns_and_are_reported():
    """A stable column set matters: a not-yet-listed name must not change the panel's shape."""
    empty = bars(SESSIONS).iloc[0:0]
    p = build_panel(
        {"E": (empty, empty_actions()), "X": (bars(SESSIONS), empty_actions())}, SESSIONS
    )
    assert p.empty == ["E"] and list(p.close.columns) == ["E", "X"]
    assert p.close["E"].isna().all() and not p.observed["E"].any() and (p.dividends["E"] == 0).all()
    assert p.eligible(SESSIONS[0], SESSIONS[-1], 0.5) == ["X"]
