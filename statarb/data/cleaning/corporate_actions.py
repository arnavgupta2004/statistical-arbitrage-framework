"""Corporate-action arithmetic, split out so each identity can be tested on its own.

Conventions (vendor basis, see ``sources/base.py``)
---------------------------------------------------
Let ``k_s`` be the split ratio (new/old shares) of a split with ex-date ``s`` and ``C_t`` the
vendor's split-adjusted close.

* As-traded price:           ``raw_t = C_t * prod(k_s : s > t)``  (all splits through the download)
* **As-of-T view** (causal): the series a vendor would have delivered on day T is
  ``C_t^(T) = C_t * F_T`` with ``F_T = prod(k_s : s > T)`` -- a constant per ticker.  Returns are
  unchanged by ``F_T`` but *levels* are not, so any level-based statistic (a spread ``Y - b X``, a
  z-score of a price ratio) computed on today's vendor basis silently depends on splits that had
  not happened yet.  ``rebase_to_as_of`` removes exactly that dependence.
* Total return with dividends reinvested at the ex-date close:
      ``r_t = (C_t + D_t) / C_{t-1} - 1``,   ``I_t = C_t * G_t``,
      ``G_t = prod(1 + D_u / C_u : u <= t)``
  so ``I_t / I_{t-1} = 1 + r_t``.  ``G`` depends only on dividends up to ``t``: a dividend-adjusted
  price *anchored at the analysis date* (``I_t / G_T * ...``) uses no future dividend, unlike the
  vendor's ``Adj Close`` whose whole history is rescaled by every later dividend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_LEVEL_COLUMNS = ["open", "high", "low", "close"]


def splits_of(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[actions["kind"] == "split"] if len(actions) else actions


def dividends_of(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[actions["kind"] == "dividend"] if len(actions) else actions


def future_split_factor(actions: pd.DataFrame, as_of: pd.Timestamp) -> float:
    """``F_T``: product of split ratios with ex-date strictly after ``as_of``."""
    s = splits_of(actions)
    if not len(s):
        return 1.0
    return float(np.prod(s.loc[s["date"] > as_of, "value"].to_numpy()))


def rebase_to_as_of(
    prices: pd.DataFrame, actions: pd.DataFrame, as_of: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the price/action history exactly as a vendor would have delivered it on ``as_of``.

    Rows and actions dated after ``as_of`` are removed and the remaining values are re-expressed
    on the split basis of that day.  Corrects for splits only; retroactive *dividend* restatements
    by the vendor are not visible in stored history and are a documented limitation.
    """
    as_of = pd.Timestamp(as_of)
    factor = future_split_factor(actions, as_of)
    p = prices[prices.index <= as_of].copy()
    for col in PRICE_LEVEL_COLUMNS:
        if col in p.columns:
            p[col] = p[col] * factor
    if "volume" in p.columns:
        p["volume"] = p["volume"] / factor
    a = actions[actions["date"] <= as_of].copy() if len(actions) else actions.copy()
    if len(a):
        is_div = a["kind"] == "dividend"
        a.loc[is_div, "value"] = a.loc[is_div, "value"] * factor
    return p, a


def as_traded_close(close: pd.Series, actions: pd.DataFrame) -> pd.Series:
    """Reconstruct the price that printed on each day: ``C_t * prod(k_s : s > t)``."""
    s = splits_of(actions)
    if not len(s):
        return close.copy()
    ratios = s.sort_values("date")
    # factor for row t = product of ratios of splits strictly after t
    later = np.ones(len(close))
    for d, k in zip(ratios["date"], ratios["value"], strict=True):
        later = later * np.where(close.index < d, k, 1.0)
    return close * later


def assign_dividends_to_rows(row_dates: pd.DatetimeIndex, dividends: pd.DataFrame) -> pd.Series:
    """Cash dividend attached to the first stored row on/after each ex-date.

    If the ex-date bar itself is missing the dividend is paid out on the next observed bar, so it
    is never lost; a dividend whose ex-date is past the last row is not yet knowable and is dropped.
    """
    out = pd.Series(0.0, index=row_dates)
    if not len(dividends) or not len(row_dates):
        return out
    pos = row_dates.searchsorted(dividends["date"].to_numpy(), side="left")
    for p, amount in zip(pos, dividends["value"].to_numpy(), strict=True):
        if p < len(row_dates):
            out.iloc[p] += amount
    return out


def dividend_growth(close: pd.Series, dividends_on_rows: pd.Series) -> pd.Series:
    """``G_t = prod(1 + D_u / C_u : u <= t)`` -- cumulative dividend reinvestment factor."""
    g = 1.0 + dividends_on_rows.reindex(close.index).fillna(0.0) / close
    return g.cumprod()
