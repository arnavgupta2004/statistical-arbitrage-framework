"""Trading calendar (exchange_calendars) with a causal notion of "last finalised session"."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd


def to_naive_dates(index) -> pd.DatetimeIndex:
    """Convert any datetime index to naive, midnight-normalised, ns-resolution session labels.

    Vendor timestamps are exchange-local midnight (Yahoo) or plain dates (CSV).  Dropping the
    timezone *without* converting keeps the local calendar date, which is what a session label is.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize().astype("datetime64[ns]")


@lru_cache(maxsize=8)
def _load_calendar(name: str, start: str, end: str):
    import exchange_calendars as xc

    return xc.get_calendar(name, start=start, end=end)


class TradingCalendar:
    def __init__(self, name: str = "XNYS", start: str = "1995-01-01", end: str | None = None):
        end = end or (pd.Timestamp.now() + pd.DateOffset(years=1)).strftime("%Y-%m-01")
        self.name = name
        self._cal = _load_calendar(name, start, end)

    def sessions(self, start, end) -> pd.DatetimeIndex:
        """Sessions in [start, end], inclusive, as naive dates."""
        return to_naive_dates(self._cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end)))

    def is_session(self, dates) -> pd.Series:
        idx = to_naive_dates(dates)
        if len(idx) == 0:
            return pd.Series(dtype=bool)
        known = self.sessions(idx.min(), idx.max())
        return pd.Series(idx.isin(known), index=idx)

    def last_complete_session(self, now: pd.Timestamp, buffer_hours: float = 2.0) -> pd.Timestamp:
        """Latest session whose closing bell was at least ``buffer_hours`` before ``now``.

        A bar downloaded during or right after the session is provisional (vendors revise the
        final print and volume), so the pipeline never stores it.  ``now`` must be tz-aware UTC.
        """
        now = pd.Timestamp(now)
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        close = self._cal.previous_close(now - pd.Timedelta(hours=buffer_hours))
        return pd.Timestamp(self._cal.minute_to_session(close, direction="previous")).tz_localize(
            None
        )
