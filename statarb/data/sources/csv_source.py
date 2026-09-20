"""Directory of manually downloaded CSV files (e.g. Stooq exports) as a data source.

Stooq's bulk CSV endpoint sits behind an interactive browser check, so it cannot be scripted
legitimately.  Files downloaded by hand from stooq.com can be dropped into a directory and used
either as a primary source or, more usefully, as an *independent cross-check* of Yahoo
(``cleaning.validation.compare_sources``).

Expected layout: ``<dir>/<TICKER>.csv`` with columns ``Date,Open,High,Low,Close,Volume`` (Stooq's
header).  Optional ``<dir>/<TICKER>.actions.csv`` with ``date,kind,value``.  Stooq prices are
split-adjusted and not dividend-adjusted, which matches the vendor basis in ``sources/base.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from statarb.data.sources.base import (
    PRICE_COLUMNS,
    RawHistory,
    SourceError,
    empty_actions,
    empty_prices,
)


class CsvDirectorySource:
    name = "csv"

    def __init__(self, directory: str | Path, suffix: str = ""):
        self.directory = Path(directory)
        self.suffix = suffix  # e.g. ".us" for Stooq's "aapl.us.csv"

    def _path(self, ticker: str) -> Path:
        stem = ticker.lower() if self.suffix else ticker
        return self.directory / f"{stem}{self.suffix}.csv"

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> RawHistory:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        path = self._path(ticker)
        if not path.exists():
            return RawHistory(ticker, empty_prices(), empty_actions(), self.name, now)
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise SourceError(f"cannot parse {path}: {exc}") from exc
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in ("date", *PRICE_COLUMNS) if c not in df.columns]
        if missing:
            raise SourceError(f"{path} lacks columns {missing}")
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        prices = df.set_index("date")[PRICE_COLUMNS].astype("float64")
        prices = prices[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))]

        actions = empty_actions()
        apath = path.with_suffix(".actions.csv")
        if apath.exists():
            a = pd.read_csv(apath, parse_dates=["date"])
            a["date"] = a["date"].astype("datetime64[ns]")
            a["value"] = a["value"].astype("float64")
            actions = a[
                (a["date"] >= pd.Timestamp(start)) & (a["date"] <= pd.Timestamp(end))
            ].reset_index(drop=True)
        return RawHistory(ticker, prices, actions, self.name, now)
