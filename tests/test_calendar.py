from __future__ import annotations

import pandas as pd
import pytest

from statarb.data.cleaning.calendar import to_naive_dates


def test_sessions_skip_weekends_and_holidays(calendar):
    s = calendar.sessions("2024-07-01", "2024-07-08")
    assert list(s.strftime("%Y-%m-%d")) == [
        "2024-07-01",
        "2024-07-02",
        "2024-07-03",
        "2024-07-05",
        "2024-07-08",
    ]  # July 4th closed, and the July 3rd session is a (shortened) session


def test_special_closures_are_known(calendar):
    # 2025-01-09: national day of mourning, NYSE closed on a Thursday
    assert pd.Timestamp("2025-01-09") not in calendar.sessions("2025-01-06", "2025-01-10")
    # 2012-10-29/30: Hurricane Sandy
    s = calendar.sessions("2012-10-26", "2012-11-01")
    assert pd.Timestamp("2012-10-29") not in s and pd.Timestamp("2012-10-30") not in s


def test_is_session(calendar):
    flags = calendar.is_session(pd.DatetimeIndex(["2024-07-03", "2024-07-04", "2024-07-06"]))
    assert flags.tolist() == [True, False, False]


@pytest.mark.parametrize(
    "now_utc, expected",
    [
        ("2026-09-18 19:30", "2026-09-17"),  # 15:30 NY: session still open
        (
            "2026-09-18 20:30",
            "2026-09-17",
        ),  # bell rang 30 min ago: bar still provisional (2h buffer)
        ("2026-09-18 22:01", "2026-09-18"),  # 2h+ after the bell: final
        ("2026-09-19 15:00", "2026-09-18"),  # Saturday -> Friday
        ("2026-09-21 12:00", "2026-09-18"),  # Monday morning before the open -> Friday
    ],
)
def test_last_complete_session(calendar, now_utc, expected):
    got = calendar.last_complete_session(pd.Timestamp(now_utc, tz="UTC"), buffer_hours=2.0)
    assert got == pd.Timestamp(expected)


def test_last_complete_session_requires_timezone(calendar):
    with pytest.raises(ValueError):
        calendar.last_complete_session(pd.Timestamp("2026-09-18 22:00"))


def test_to_naive_dates_keeps_the_local_calendar_date():
    idx = pd.DatetimeIndex(["2020-08-31 00:00:00-04:00"]).tz_convert("America/New_York")
    assert to_naive_dates(idx)[0] == pd.Timestamp("2020-08-31")  # not shifted to the UTC date
