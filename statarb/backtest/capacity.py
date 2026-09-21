"""Capacity of a strategy from its cost components, without re-running it.

In the cost model (``backtest/costs.py``) trades are fixed fractions of capital, so as capital ``C``
changes: spread, commission and borrow (per unit of capital) do not move, **market impact per
unit of capital scales with the square root of ``C``** (``Y sigma sqrt(Q/ADV)``, ``Q = w C``), and
the participation rate ``Q / ADV`` scales linearly.  A daily net-return series at any capital, and
the capital at which the mean net return reaches zero, therefore follow exactly from the components
saved at one reference capital.  (The exactness is tested against a re-run at another capital.)

Multipliers let the analysis ask "what would spreads / impact / fees have to be?": they scale a
component after the capital adjustment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = ("cost_spread", "cost_commission", "cost_impact", "cost_borrow")


def net_at_capital(
    daily: pd.DataFrame,
    capital: float,
    base_capital: float,
    spread_mult: float = 1.0,
    commission_mult: float = 1.0,
    impact_mult: float = 1.0,
    borrow_mult: float = 1.0,
) -> pd.Series:
    """Daily net return at ``capital`` from a frame with ``pnl`` and the four cost components."""
    scale = np.sqrt(capital / base_capital)
    cost = (
        spread_mult * daily["cost_spread"]
        + commission_mult * daily["cost_commission"]
        + impact_mult * scale * daily["cost_impact"]
        + borrow_mult * daily["cost_borrow"]
    )
    return daily["pnl"] - cost


def participation_breach_share(
    participation: pd.Series, capital: float, base_capital: float, limit: float = 0.10
) -> float:
    """Share of days on which some trade exceeds ``limit`` of ADV at ``capital``."""
    return float((participation * (capital / base_capital) > limit).mean())


def break_even_capital(
    daily: pd.DataFrame,
    base_capital: float,
    spread_mult: float = 1.0,
    commission_mult: float = 1.0,
    impact_mult: float = 1.0,
    borrow_mult: float = 1.0,
) -> float:
    """Capital at which the *mean* net return is zero: 0 if it is negative even without impact,
    ``inf`` if impact is zero and the rest of the costs leave a profit."""
    gross = daily["pnl"].mean()
    fixed = (
        spread_mult * daily["cost_spread"].mean()
        + commission_mult * daily["cost_commission"].mean()
        + borrow_mult * daily["cost_borrow"].mean()
    )
    impact = impact_mult * daily["cost_impact"].mean()
    room = gross - fixed
    if room <= 0:
        return 0.0
    if impact <= 0:
        return float("inf")
    return float(base_capital * (room / impact) ** 2)
