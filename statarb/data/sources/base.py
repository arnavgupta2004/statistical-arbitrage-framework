"""Vendor-neutral contract for raw daily history.

Vendor basis (what every source must deliver, and what the store keeps)
----------------------------------------------------------------------
* ``open/high/low/close``: **split-adjusted, not dividend-adjusted**, relative to the vendor's
  download date.  A split that happens after the download rescales all earlier rows on the next
  download; the refresh logic detects that (see ``download/refresh.py``).
* ``volume``: split-adjusted in the opposite direction, so ``close * volume`` (dollar volume) is
  invariant to splits.
* ``actions``: ``dividend`` rows (cash per share on the same split-adjusted basis) and ``split``
  rows (ratio = new shares / old shares, so a 4-for-1 split is 4.0 and a 1-for-10 reverse split
  is 0.1).
* ``adj_close`` (optional): the vendor's own dividend-adjusted close.  It is used only to
  cross-check our adjustment at ingest and is never stored: it is rescaled retroactively by every
  future dividend, so a stored copy would silently mix bases across incremental downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
ACTION_COLUMNS = ["date", "kind", "value"]


class SourceError(Exception):
    """Permanent failure for this request (bad symbol format, parse failure, ...)."""


class TransientSourceError(SourceError):
    """Failure worth retrying with back-off (rate limit, timeout, dropped connection)."""


def empty_prices() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], name="date").astype("datetime64[ns]")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in PRICE_COLUMNS}, index=idx)


def empty_actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "kind": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )


@dataclass(frozen=True)
class RawHistory:
    ticker: str
    prices: (
        pd.DataFrame
    )  # index: naive session dates named "date"; columns PRICE_COLUMNS [+ adj_close]
    actions: pd.DataFrame  # columns ACTION_COLUMNS
    source: str
    fetched_at: pd.Timestamp  # naive UTC


class DataSource(Protocol):
    name: str

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> RawHistory:
        """History for sessions in [start, end] (both inclusive).

        An empty result is *not* an error: it is the normal answer for a range with no new
        sessions and for a symbol the vendor does not know.  The caller decides which it is.
        """
        ...
