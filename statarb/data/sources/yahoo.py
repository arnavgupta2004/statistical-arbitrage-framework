"""Yahoo Finance daily history via ``yfinance``.

Licensing: Yahoo's public endpoints are intended for personal use; this project is
non-commercial research and keeps downloaded data in a local, git-ignored store.  Do not
redistribute the store.  yfinance is an unofficial scraper, so the pipeline treats every
response as untrusted and validates it (``cleaning/``).

Yahoo's ``Close``/``Open``/``High``/``Low`` are split-adjusted; ``Adj Close`` additionally folds
in dividends *as of the download date*; ``Volume`` is scaled inversely to splits (see
``sources/base.py`` for the contract this class fulfils).
"""

from __future__ import annotations

import pandas as pd

from statarb.data.cleaning.calendar import to_naive_dates
from statarb.data.sources.base import (
    PRICE_COLUMNS,
    RawHistory,
    SourceError,
    TransientSourceError,
    empty_actions,
    empty_prices,
)

_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


class YahooSource:
    name = "yahoo"

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> RawHistory:
        import yfinance as yf
        from yfinance import exceptions as yfe

        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        try:
            # yfinance's `end` is exclusive; its day boundaries are exchange-local, which can leak
            # the neighbouring session at either edge, so we re-filter to [start, end] below.
            raw = yf.Ticker(ticker).history(
                start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=False,
                actions=True,
                raise_errors=True,
            )
        except yfe.YFRateLimitError as exc:
            raise TransientSourceError(f"rate limited: {exc}") from exc
        except (yfe.YFPricesMissingError, yfe.YFTickerMissingError, yfe.YFTzMissingError):
            return RawHistory(ticker, empty_prices(), empty_actions(), self.name, now)
        except Exception as exc:  # network / parsing errors from requests, curl_cffi, json
            name = type(exc).__name__
            if any(k in name for k in ("Timeout", "Connection", "HTTP", "SSL", "Curl", "Chunked")):
                raise TransientSourceError(f"{name}: {exc}") from exc
            raise SourceError(f"{name}: {exc}") from exc

        if raw is None or raw.empty:
            return RawHistory(ticker, empty_prices(), empty_actions(), self.name, now)

        idx = to_naive_dates(raw.index)
        df = raw.rename(columns=_RENAME)
        keep = [*PRICE_COLUMNS, "adj_close"]
        prices = df[[c for c in keep if c in df.columns]].copy()
        prices.index = pd.DatetimeIndex(idx, name="date")
        in_range = (prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))
        prices = prices[in_range]

        actions = _extract_actions(raw, idx)
        if len(actions):
            actions = actions[
                (actions["date"] >= pd.Timestamp(start)) & (actions["date"] <= pd.Timestamp(end))
            ]
        return RawHistory(ticker, prices, actions.reset_index(drop=True), self.name, now)


def _extract_actions(raw: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    if "Dividends" in raw.columns:
        m = raw["Dividends"].fillna(0).to_numpy() > 0
        rows.append(
            pd.DataFrame(
                {"date": idx[m], "kind": "dividend", "value": raw["Dividends"].to_numpy()[m]}
            )
        )
    if "Stock Splits" in raw.columns:
        m = raw["Stock Splits"].fillna(0).to_numpy() > 0
        rows.append(
            pd.DataFrame(
                {"date": idx[m], "kind": "split", "value": raw["Stock Splits"].to_numpy()[m]}
            )
        )
    if not rows:
        return empty_actions()
    out = pd.concat(rows, ignore_index=True)
    out["date"] = out["date"].astype("datetime64[ns]")
    out["value"] = out["value"].astype("float64")
    return out.sort_values(["date", "kind"]).reset_index(drop=True)
