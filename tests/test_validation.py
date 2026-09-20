"""Every audit must fire on the defect it is named for and stay silent on clean data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.config import ValidationConfig
from statarb.data.cleaning.issues import Severity, issues_to_frame, summarize
from statarb.data.cleaning.sanitize import sanitize_history
from statarb.data.cleaning.validation import audit_history, compare_sources
from statarb.data.sources.base import empty_actions
from statarb.data.sources.synthetic import SyntheticSource

from .conftest import make_world, ts

CFG = ValidationConfig()


@pytest.fixture
def clean(calendar):
    """A defect-free BBB history (dividends, no splits) and the calendar sessions around it."""
    raw = SyntheticSource(make_world()).fetch("BBB", ts("2015-01-01"), ts("2016-12-30"))
    prices = raw.prices.drop(columns="adj_close")
    return prices, raw.actions, calendar.sessions("2015-01-01", "2017-01-31")


def codes(issues):
    return {i.code for i in issues}


def test_clean_data_raises_no_warning_or_error(clean):
    prices, actions, sessions = clean
    issues = audit_history(
        "BBB", prices, actions, sessions, CFG, last_final_session=ts("2016-12-30")
    )
    assert [i for i in issues if i.severity != Severity.INFO] == []


def test_correctly_split_adjusted_history_is_not_flagged(calendar):
    """False-positive guard: AAA has a 2-for-1; the vendor's adjusted series is continuous."""
    raw = SyntheticSource(make_world()).fetch("AAA", ts("2015-01-01"), ts("2016-12-30"))
    issues = audit_history(
        "AAA",
        raw.prices.drop(columns="adj_close"),
        raw.actions,
        calendar.sessions("2015-01-01", "2017-01-31"),
        CFG,
    )
    assert not codes(issues) & {"SPLIT_UNADJUSTED", "POSSIBLE_UNRECORDED_SPLIT", "EXTREME_RETURN"}


# ---- sanitize -------------------------------------------------------------------------------
def test_sanitize_drops_duplicates_keeping_the_last_row(clean):
    prices, actions, _ = clean
    dup = pd.concat([prices, prices.iloc[[10]].assign(close=prices["close"].iloc[10] * 1.01)])
    p, _, issues = sanitize_history("BBB", dup, actions)
    assert "DUPLICATE_DATES" in codes(issues) and len(p) == len(prices)
    assert p["close"].iloc[10] == pytest.approx(prices["close"].iloc[10] * 1.01)  # last one wins


def test_sanitize_sorts_and_reports_unsorted_index(clean):
    prices, actions, _ = clean
    p, _, issues = sanitize_history("BBB", prices.iloc[::-1], actions)
    assert "UNSORTED_INDEX" in codes(issues) and p.index.is_monotonic_increasing


def test_sanitize_drops_rows_without_a_valid_close(clean):
    prices, actions, _ = clean
    bad = prices.copy()
    bad.iloc[5, bad.columns.get_loc("close")] = np.nan
    bad.iloc[6, bad.columns.get_loc("close")] = 0.0
    bad.iloc[7, bad.columns.get_loc("close")] = -3.0
    p, _, issues = sanitize_history("BBB", bad, actions)
    assert len(p) == len(prices) - 3
    assert next(i for i in issues if i.code == "INVALID_CLOSE").count == 3


def test_sanitize_handles_timezone_aware_vendor_index(clean):
    prices, actions, _ = clean
    tz = prices.copy()
    tz.index = tz.index.tz_localize("America/New_York")
    p, _, _ = sanitize_history("BBB", tz, actions)
    assert p.index.equals(prices.index) and p.index.tz is None


def test_sanitize_drops_invalid_actions():
    prices = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": [1.0, 1.0], "volume": 1.0},
        index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"], name="date"),
    )
    acts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 3),
            "kind": ["dividend", "split", "bogus"],
            "value": [-1.0, 0.0, 2.0],
        }
    )
    _, a, issues = sanitize_history("X", prices, acts)
    assert a.empty and "INVALID_ACTION" in codes(issues)


# ---- audit ----------------------------------------------------------------------------------
def test_missing_sessions_warning_then_error_when_the_gap_is_long(clean):
    prices, actions, sessions = clean
    few = prices.drop(prices.index[[20, 50, 90]])
    issue = next(
        i for i in audit_history("BBB", few, actions, sessions, CFG) if i.code == "MISSING_SESSIONS"
    )
    assert issue.count == 3 and issue.severity == Severity.WARNING
    gap = prices.drop(prices.index[100:108])  # 8 consecutive sessions > max_gap_sessions=5
    issue = next(
        i for i in audit_history("BBB", gap, actions, sessions, CFG) if i.code == "MISSING_SESSIONS"
    )
    assert issue.severity == Severity.ERROR


def test_calendar_mismatch_on_weekend_rows(clean):
    prices, actions, sessions = clean
    sat = ts("2015-06-06")  # a Saturday
    bad = pd.concat(
        [prices, prices.iloc[[100]].rename(index={prices.index[100]: sat})]
    ).sort_index()
    issue = next(
        i
        for i in audit_history("BBB", bad, actions, sessions, CFG)
        if i.code == "CALENDAR_MISMATCH"
    )
    assert issue.count == 1 and issue.first_date == sat


def test_future_dates_flagged_as_provisional(clean):
    prices, actions, sessions = clean
    issue = next(
        i
        for i in audit_history(
            "BBB", prices, actions, sessions, CFG, last_final_session=ts("2016-12-28")
        )
        if i.code == "FUTURE_DATES"
    )
    assert issue.count == 2  # 12-29 and 12-30


def test_stale_price_run(clean):
    prices, actions, sessions = clean
    bad = prices.copy()
    bad.iloc[200:206, bad.columns.get_loc("close")] = bad["close"].iloc[200]  # 6 identical closes
    issue = next(
        i for i in audit_history("BBB", bad, actions, sessions, CFG) if i.code == "STALE_PRICE"
    )
    assert issue.count == 6
    ok = prices.copy()
    ok.iloc[200:203, ok.columns.get_loc("close")] = ok["close"].iloc[
        200
    ]  # 3 in a row: below threshold
    assert "STALE_PRICE" not in codes(audit_history("BBB", ok, actions, sessions, CFG))


def test_extreme_return_flagged(clean):
    prices, actions, sessions = clean
    bad = prices.copy()
    bad.iloc[150:, bad.columns.get_loc("close")] *= 1.9  # +90% one-day jump that persists
    issue = next(
        i for i in audit_history("BBB", bad, actions, sessions, CFG) if i.code == "EXTREME_RETURN"
    )
    assert issue.count == 1 and issue.first_date == prices.index[150]


def _unadjust_split(prices, at_position, ratio):
    """What a vendor that forgot to split-adjust would deliver: pre-split closes are ratio x higher."""
    bad = prices.copy()
    bad.iloc[:at_position, bad.columns.get_indexer(["open", "high", "low", "close"])] *= ratio
    return bad


def test_unadjusted_recorded_split_is_an_error(clean):
    prices, _, sessions = clean
    pos, k = 250, 4.0
    bad = _unadjust_split(prices, pos, k)
    acts = pd.DataFrame({"date": [prices.index[pos]], "kind": ["split"], "value": [k]})
    issue = next(
        i for i in audit_history("BBB", bad, acts, sessions, CFG) if i.code == "SPLIT_UNADJUSTED"
    )
    assert issue.severity == Severity.ERROR and issue.first_date == prices.index[pos]


def test_unrecorded_split_is_a_warning_and_not_double_reported(clean):
    prices, _, sessions = clean
    bad = _unadjust_split(prices, 250, 3.0)
    got = audit_history("BBB", bad, empty_actions(), sessions, CFG)
    assert "POSSIBLE_UNRECORDED_SPLIT" in codes(got)
    assert "SPLIT_UNADJUSTED" not in codes(got)


def test_reverse_split_left_unadjusted_is_detected(clean):
    prices, _, sessions = clean
    bad = _unadjust_split(
        prices, 250, 0.1
    )  # 1-for-10 reverse split forgotten: pre-split prices 10x lower
    assert "POSSIBLE_UNRECORDED_SPLIT" in codes(
        audit_history("BBB", bad, empty_actions(), sessions, CFG)
    )


def test_ohlc_inconsistency(clean):
    prices, actions, sessions = clean
    bad = prices.copy()
    bad.iloc[30, bad.columns.get_loc("high")] = bad["close"].iloc[30] * 0.9
    assert "OHLC_INCONSISTENT" in codes(audit_history("BBB", bad, actions, sessions, CFG))


def test_empty_history_is_an_error(clean):
    _, actions, sessions = clean
    got = audit_history(
        "ZZZ",
        pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).astype(float),
        actions,
        sessions,
        CFG,
    )
    assert [i.code for i in got] == ["NO_DATA"]


def test_issue_table_and_summary(clean):
    prices, actions, sessions = clean
    bad = prices.drop(prices.index[[20, 50]])
    issues = audit_history("BBB", bad, actions, sessions, CFG)
    df = issues_to_frame(issues)
    assert list(df.columns)[:3] == ["ticker", "code", "severity"]
    assert summarize(issues).loc[0, "observations"] == 2
    assert issues_to_frame([]).empty and summarize([]).empty


def test_compare_sources_agreement_and_disagreement(clean):
    prices, _, _ = clean
    same = compare_sources(prices, prices)
    assert same["share_differing"] == 0 and same["overlap"] == len(prices)
    other = prices.copy()
    other.iloc[::10, other.columns.get_loc("close")] *= 1.02
    diff = compare_sources(prices, other)
    assert diff["share_differing"] == pytest.approx(np.ceil(len(prices) / 10) / len(prices))
    assert compare_sources(prices, prices.iloc[0:0])["overlap"] == 0


# ---- non-ordinary distributions and split-ratio matching ------------------------------------
def test_large_dividend_is_flagged_but_ordinary_ones_are_not(clean):
    prices, actions, sessions = clean
    assert "LARGE_DIVIDEND" not in codes(audit_history("BBB", prices, actions, sessions, CFG))
    day = prices.index[200]
    special = pd.DataFrame(
        {"date": [day], "kind": ["dividend"], "value": [0.35 * prices["close"].iloc[199]]}
    )
    issue = next(
        i
        for i in audit_history("BBB", prices, pd.concat([actions, special]), sessions, CFG)
        if i.code == "LARGE_DIVIDEND"
    )
    assert issue.count == 1 and issue.first_date == day and "35" in issue.detail


def test_a_genuine_crash_near_a_split_ratio_is_not_mistaken_for_a_split(clean):
    """-82% is not a 5-for-1 split (which would be -80%): the matcher must be tight in log space."""
    prices, _, sessions = clean
    crash = prices.copy()
    crash.iloc[300:, crash.columns.get_indexer(["open", "high", "low", "close"])] *= 0.18
    assert "POSSIBLE_UNRECORDED_SPLIT" not in codes(
        audit_history("BBB", crash, empty_actions(), sessions, CFG)
    )
    exact = prices.copy()
    exact.iloc[300:, exact.columns.get_indexer(["open", "high", "low", "close"])] *= 0.2
    assert "POSSIBLE_UNRECORDED_SPLIT" in codes(
        audit_history("BBB", exact, empty_actions(), sessions, CFG)
    )
