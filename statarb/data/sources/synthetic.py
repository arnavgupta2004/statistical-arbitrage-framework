"""Deterministic synthetic vendor with a controllable "vendor date".

Purpose: exercise the pipeline offline and, more importantly, make *causality* testable.  A
latent world is generated once (economic prices ``E_t`` on the final split basis, plus known splits
and dividends).  A vendor whose clock reads ``vendor_date = V`` delivers exactly what a real vendor
would have delivered on day V:

    close_t^(V)  = E_t * prod(ratio_s for world splits with s > V)      (only splits <= V applied)
    volume_t^(V) = Ev_t / prod(ratio_s for world splits with s > V)
    dividends^(V) = D_s^E * prod(ratio for splits > V), for dividends with ex-date <= V

Two vendors over the same world with different ``vendor_date`` therefore differ *only* in the
retroactive rescaling that real vendors apply, which is what point-in-time views must undo.
Moving ``vendor_date`` forward on one instance simulates the next day's download.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from statarb.data.cleaning.calendar import TradingCalendar
from statarb.data.sources.base import RawHistory, empty_actions, empty_prices


@dataclass
class SyntheticWorld:
    tickers: list[str]
    start: str = "2015-01-02"
    end: str = "2020-12-31"
    seed: int = 0
    mu: float = 0.0003
    sigma: float = 0.015
    price0: float = 100.0
    splits: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    dividends: dict[str, list[tuple[str, float]]] = field(default_factory=dict)  # E-basis amounts
    lifespans: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    calendar: str = "XNYS"


class SyntheticSource:
    name = "synthetic"

    def __init__(self, world: SyntheticWorld, vendor_date: str | pd.Timestamp | None = None):
        self.world = world
        self.sessions = TradingCalendar(world.calendar).sessions(world.start, world.end)
        self.vendor_date = (
            pd.Timestamp(vendor_date) if vendor_date is not None else self.sessions[-1]
        )
        self.calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
        self._cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    # ---- latent world -------------------------------------------------------------------
    def _generate(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if ticker in self._cache:
            return self._cache[ticker]
        w = self.world
        rng = np.random.default_rng([w.seed, zlib.crc32(ticker.encode())])
        n = len(self.sessions)
        logret = rng.normal(w.mu, w.sigma, n)
        close = w.price0 * np.exp(np.cumsum(logret))
        prev = np.concatenate([[w.price0], close[:-1]])
        open_ = prev * np.exp(rng.normal(0, w.sigma / 4, n))
        span = np.abs(rng.normal(0, w.sigma / 2, n))
        high = np.maximum(open_, close) * (1 + span)
        low = np.minimum(open_, close) * (1 - span)
        volume = np.exp(rng.normal(14.0, 0.4, n)).round()
        prices = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.DatetimeIndex(self.sessions, name="date"),
        )
        lo, hi = w.lifespans.get(ticker, (None, None))
        if lo is not None:
            prices = prices[prices.index >= pd.Timestamp(lo)]
        if hi is not None:
            prices = prices[prices.index <= pd.Timestamp(hi)]

        rows = []
        for d, ratio in w.splits.get(ticker, []):
            self._require_session(d)
            rows.append((pd.Timestamp(d), "split", float(ratio)))
        for d, amount in w.dividends.get(ticker, []):
            self._require_session(d)
            rows.append((pd.Timestamp(d), "dividend", float(amount)))
        actions = pd.DataFrame(rows, columns=["date", "kind", "value"]) if rows else empty_actions()
        actions["date"] = actions["date"].astype("datetime64[ns]")
        self._cache[ticker] = (prices, actions.sort_values(["date", "kind"]).reset_index(drop=True))
        return self._cache[ticker]

    def _require_session(self, d: str) -> None:
        if pd.Timestamp(d) not in self.sessions:
            raise ValueError(f"{d} is not a trading session")

    # ---- vendor view --------------------------------------------------------------------
    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> RawHistory:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        self.calls.append((ticker, start, end))
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        if ticker not in self.world.tickers:
            return RawHistory(ticker, empty_prices(), empty_actions(), self.name, now)

        prices_e, actions_e = self._generate(ticker)
        v = self.vendor_date
        splits = actions_e[actions_e["kind"] == "split"]
        f_v = float(np.prod(splits.loc[splits["date"] > v, "value"])) if len(splits) else 1.0

        p = prices_e[prices_e.index <= v].copy()
        for c in ("open", "high", "low", "close"):
            p[c] = p[c] * f_v
        p["volume"] = p["volume"] / f_v

        a = actions_e[actions_e["date"] <= v].copy()
        a.loc[a["kind"] == "dividend", "value"] *= f_v

        # Yahoo-style dividend adjustment: earlier closes scaled by (1 - D / previous close)
        divs = a[a["kind"] == "dividend"]
        adj = pd.Series(1.0, index=p.index)
        for d, amount in zip(divs["date"], divs["value"], strict=True):
            pos = p.index.searchsorted(d)
            if 0 < pos < len(p):
                adj.iloc[:pos] *= 1.0 - amount / p["close"].iloc[pos - 1]
        p["adj_close"] = p["close"] * adj

        mask = (p.index >= start) & (p.index <= end)
        amask = (a["date"] >= start) & (a["date"] <= end)
        return RawHistory(ticker, p[mask], a[amask].reset_index(drop=True), self.name, now)
