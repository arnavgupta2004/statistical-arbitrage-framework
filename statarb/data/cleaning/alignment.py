"""Alignment of per-ticker histories onto the trading calendar.

Policy: **no imputation**.  A ticker with no bar on a session has NaN there and ``observed`` is
False; returns are defined only between two *consecutive observed sessions*, so a gap never turns
into a fabricated multi-day "daily" return, and prices are never forward-filled.  Whoever consumes
the panel (screening, PCA, the backtester) decides explicitly how to treat missing observations.

Causality: ``build_panel`` is a pure function of the histories passed in.  Point-in-time
correctness comes from calling it on histories rebased to an ``as_of`` date
(``corporate_actions.rebase_to_as_of``) -- ``DataPipeline.panel`` does this.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from statarb.data.cleaning.corporate_actions import (
    assign_dividends_to_rows,
    dividend_growth,
    dividends_of,
)


@dataclass
class AlignedPanel:
    """Session-by-ticker panels sharing one index/columns.  Vendor (split-adjusted) basis."""

    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    dividends: pd.DataFrame  # cash dividend per share on the ex-date row (0 elsewhere)
    ret: pd.DataFrame  # total return; NaN unless both this and the previous session are observed
    observed: pd.DataFrame  # bool: a stored bar exists on this session
    growth: pd.DataFrame  # G_t, cumulative dividend reinvestment factor (see corporate_actions)
    as_of: pd.Timestamp | None = None
    dropped: dict[str, str] = field(default_factory=dict)  # requested but unusable (set by callers)
    empty: list[str] = field(default_factory=list)  # kept as all-NaN columns: no bar in the window

    @property
    def dollar_volume(self) -> pd.DataFrame:
        """close * volume: invariant to splits, the input to ADV and impact models."""
        return self.close * self.volume

    def tr_close(self, anchor: pd.Timestamp | None = None) -> pd.DataFrame:
        """Dividend-adjusted price, back-adjusted from ``anchor`` (default: the last session).

        ``I_t = C_t * G_t / G_anchor``.  Uses only dividends up to ``anchor``; pass an anchor no
        later than the information date of the calling analysis.
        """
        anchor = pd.Timestamp(anchor) if anchor is not None else self.close.index[-1]
        base = self.growth.loc[:anchor].ffill().iloc[-1]
        return (self.close * self.growth / base).loc[:anchor]

    def window(self, start, end) -> AlignedPanel:
        sl = slice(pd.Timestamp(start), pd.Timestamp(end))
        return AlignedPanel(
            *(getattr(self, f).loc[sl] for f in _FRAMES),
            as_of=self.as_of,
            dropped=dict(self.dropped),
            empty=list(self.empty),
        )

    def coverage(self, start=None, end=None) -> pd.Series:
        """Observed share of the sessions in [start, end], per ticker (missing = not listed)."""
        obs = self.observed.loc[
            pd.Timestamp(start) if start else None : pd.Timestamp(end) if end else None
        ]
        return obs.mean()

    def eligible(self, start, end, min_coverage: float) -> list[str]:
        """Tickers observed on at least ``min_coverage`` of the window's sessions.

        Because coverage is measured against *all* sessions in the window, a ticker that listed
        after ``start`` or delisted before ``end`` has low coverage and is excluded automatically.
        """
        cov = self.coverage(start, end)
        return sorted(cov.index[cov >= min_coverage])


_FRAMES = ["close", "open", "high", "low", "volume", "dividends", "ret", "observed", "growth"]


def build_panel(
    histories: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    sessions: pd.DatetimeIndex,
    as_of: pd.Timestamp | None = None,
) -> AlignedPanel:
    """Assemble an ``AlignedPanel`` from ``{ticker: (prices, actions)}`` over ``sessions``."""
    cols: dict[str, dict[str, pd.Series]] = {k: {} for k in _FRAMES}
    empty: list[str] = []
    for ticker, (prices, actions) in histories.items():
        prices = prices[prices.index.isin(sessions)]
        if prices.empty:
            empty.append(ticker)
            continue
        close = prices["close"]
        div_rows = assign_dividends_to_rows(prices.index, dividends_of(actions))
        growth = dividend_growth(close, div_rows)
        obs = pd.Series(True, index=prices.index)

        cols["close"][ticker] = close
        for f in ("open", "high", "low", "volume"):
            cols[f][ticker] = prices[f]
        cols["dividends"][ticker] = div_rows
        cols["growth"][ticker] = growth
        cols["observed"][ticker] = obs

    def frame(name: str, fill=None) -> pd.DataFrame:
        df = pd.DataFrame(cols[name]).reindex(index=sessions, columns=list(histories))
        return df if fill is None else df.fillna(fill)

    close = frame("close")
    dividends = frame("dividends", 0.0)
    observed = frame("observed", False).astype(bool)
    prev = close.shift(1)
    # (C_t + D_t) / C_{t-1} - 1, defined only when both sessions have a bar
    ret = (close + dividends) / prev - 1.0
    return AlignedPanel(
        close=close,
        open=frame("open"),
        high=frame("high"),
        low=frame("low"),
        volume=frame("volume"),
        dividends=dividends,
        ret=ret.where(np.isfinite(ret)),
        observed=observed,
        growth=frame("growth"),
        as_of=as_of,
        empty=empty,
    )
