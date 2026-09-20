"""Pair-level position sizing: leg weights and volatility targeting.

Weights are in units of the pair's capital.  For a spread position ``p`` and hedge ratio ``beta``
(from a *log-price* regression) the legs are ``w_y = +p N`` and ``w_x = -p beta N``: a log-return
regression ``r_y = beta r_x`` is first-order neutral to the common factor when dollar exposures are
in the ratio ``1 : beta``.  Portfolio-level neutrality, exposure and sector limits belong to
Stage 11.

Volatility targeting sets the notional ``N`` at entry so the spread's expected daily P&L volatility
equals ``target_vol``, using the trailing standard deviation of the spread's own daily returns
(``r_y - beta r_x``) known at the entry close.  ``N`` is then frozen for the life of the trade, so
sizing itself creates no turnover, and it is capped so a quiet spread cannot lever the pair without
limit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def spread_returns(ry: pd.Series, rx: pd.Series, beta: pd.Series) -> pd.Series:
    """Return of one spread unit on day ``t``: ``ry_t - beta_{t-1} rx_t`` (hedge held overnight)."""
    return ry - beta.shift(1) * rx


def leg_weights(position, beta, notional) -> tuple[np.ndarray, np.ndarray]:
    """``(w_y, w_x)`` for a spread position, hedge ratio and notional (arrays of equal length)."""
    p = np.asarray(position, dtype=float)
    n = np.asarray(notional, dtype=float)
    b = np.nan_to_num(np.asarray(beta, dtype=float), nan=0.0)
    return p * n, -p * n * b


def entry_notional(
    position,
    spread_ret: pd.Series,
    target_vol: float | None = None,
    window: int = 60,
    max_notional: float = 3.0,
) -> np.ndarray:
    """Notional per bar: 1 (unit sizing) or vol-targeted at entry and frozen until the exit."""
    pos = np.asarray(position)
    if target_vol is None:
        return np.where(pos != 0, 1.0, 0.0)
    sigma = spread_ret.rolling(window, min_periods=window).std().to_numpy()
    n = np.zeros(len(pos))
    current = 0.0
    for t in range(len(pos)):
        if pos[t] == 0:
            current = 0.0
        elif t == 0 or pos[t - 1] == 0:
            s = sigma[t]
            current = min(max_notional, target_vol / s) if np.isfinite(s) and s > 0 else 0.0
        n[t] = current
    return n
