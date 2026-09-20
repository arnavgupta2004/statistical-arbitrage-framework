"""Frictionless P&L accounting for one pair.

Gross, before costs.  Stage 5 wraps it in a walk-forward engine and Stage 6 adds costs on the
``trade`` column.

Timing (the whole look-ahead question in one place): weights are decided at the close of ``t`` from
information up to ``t``; they earn the return of bar ``t + 1``::

    pnl_t = w_y,t-1 * ry_t + w_x,t-1 * rx_t

Rebalancing: legs are held at constant *dollar* weights, so between two closes the weights drift
with prices and must be traded back to target.  ``trade_t`` is the traded amount needed to reach
``w_t`` from the drifted ``w_{t-1} * (1 + r_t)`` (in units of pair capital), which is what
costs must be charged on: it counts entries, exits, hedge-ratio changes *and* the drift rebalancing.

``freeze_beta=True`` holds the hedge ratio at its entry value for the life of the trade instead of
re-hedging daily: less turnover, less tracking of a drifting hedge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.portfolio.construction import leg_weights


def freeze_at_entry(beta: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Hedge ratio held at its entry-close value while in a trade (0 when flat)."""
    out = np.zeros(len(beta))
    cur = 0.0
    for t in range(len(beta)):
        if position[t] == 0:
            cur = 0.0
        elif t == 0 or position[t - 1] == 0:
            cur = beta[t]
        out[t] = cur
    return out


def pair_returns(
    position, beta: pd.Series, notional, ry: pd.Series, rx: pd.Series, freeze_beta: bool = False
) -> pd.DataFrame:
    """Daily gross return on pair capital, turnover and exposure.  All inputs share ``ry.index``."""
    idx = ry.index
    pos = np.asarray(position, dtype=float)
    b = beta.reindex(idx).to_numpy(dtype=float)
    if np.any(np.isnan(b[pos != 0])):
        raise ValueError("a position is open on a date where the hedge ratio is undefined")
    if freeze_beta:
        b = freeze_at_entry(b, pos)
    w_y, w_x = leg_weights(pos, b, np.asarray(notional, dtype=float))
    r_y, r_x = ry.to_numpy(dtype=float), rx.to_numpy(dtype=float)
    held_y, held_x = np.roll(w_y, 1), np.roll(w_x, 1)
    held_y[0] = held_x[0] = 0.0
    if np.any(np.isnan(r_y[held_y != 0]) | np.isnan(r_x[held_x != 0])):
        raise ValueError(
            "a return is missing while a position is held; gaps must flatten the position"
        )
    ry0, rx0 = np.nan_to_num(r_y), np.nan_to_num(r_x)
    pnl = held_y * ry0 + held_x * rx0
    trade = np.abs(w_y - held_y * (1 + ry0)) + np.abs(w_x - held_x * (1 + rx0))
    return pd.DataFrame(
        {"pnl": pnl, "trade": trade, "gross": np.abs(w_y) + np.abs(w_x), "w_y": w_y, "w_x": w_x},
        index=idx,
    )


def run_pair(
    ly: pd.Series,
    lx: pd.Series,
    ry: pd.Series,
    rx: pd.Series,
    hedge,
    z_window: int | None,
    params,
    train_end,
    target_vol: float | None = None,
    freeze_beta: bool = False,
) -> dict:
    """The whole single-pair chain, in one place: hedge ratio -> z-score -> positions -> gross P&L.

    ``ly``/``lx`` are log prices, ``ry``/``rx`` simple total returns, all on one index.  Nothing is
    traded until the first bar after ``train_end``.  Returns the pieces so callers can inspect them.
    """
    from statarb.models.hedge_ratio import hedge_path
    from statarb.portfolio.construction import entry_notional, spread_returns
    from statarb.signals.pairs import generate_positions, trades_table
    from statarb.signals.zscore import spread_zscore

    path = hedge_path(ly, lx, hedge, train_end)
    beta = path["beta"]
    zdf = spread_zscore(ly, lx, beta, z_window)
    start = int((ly.index <= pd.Timestamp(train_end)).sum())
    res = generate_positions(zdf["z"].to_numpy(), params, trade_start=start)
    notional = entry_notional(res.position, spread_returns(ry, rx, beta), target_vol)
    out = pair_returns(res.position, beta, notional, ry, rx, freeze_beta=freeze_beta)
    return {
        "returns": out,
        "z": zdf["z"],
        "beta": beta,
        "positions": res,
        "trades": trades_table(ly.index, res, out["pnl"]),
    }
