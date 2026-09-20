"""Entry / exit / stop-loss / time-stop state machine for one spread.

Convention: ``position = +1`` is *long the spread* (long Y, short ``beta`` of X), entered when the
spread is cheap (``z < -entry``); ``-1`` is short the spread, entered when it is rich.  Let
``u = -position * z``: the stretch of the spread *against* the position (``u = |z|`` at entry).

    flat          -> enter when entry < |z| < stop  (direction = -sign(z))
    in position   -> exit  "mean"  when u <= exit    (the spread has reverted)
                  -> exit  "stop"  when u >= stop    (it kept going: relationship in doubt)
                  -> exit  "time"  after max_hold bars without either
                  -> exit  "data"  if z is undefined (a gap: never carry a position through)

* Entering beyond ``stop`` is refused: a z-score already past the stop is a break, not a
  bargain.
* After a ``stop`` or ``time`` exit the same direction is *blocked* until ``|z| <= exit`` again, so
  a
  spread that stays stretched cannot trigger an immediate re-entry (churn).
* Decisions use ``z[t]`` (information at the close of ``t``); the position it creates earns the
  returns
  from ``t`` to ``t+1`` (``backtest/pair_pnl.py`` applies that one-bar lag).
* No position is opened before ``trade_start`` (the end of the training window): the hedge ratio
  and any parameters are fixed by then, and nothing traded in-sample is ever scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REASONS = {0: "", 1: "mean", 2: "stop", 3: "time", 4: "data", 5: "end"}


@dataclass(frozen=True)
class PairParams:
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0
    max_hold: int = 60

    def __post_init__(self) -> None:
        if not 0.0 <= self.exit < self.entry < self.stop:
            raise ValueError("need 0 <= exit < entry < stop")
        if self.max_hold < 1:
            raise ValueError("max_hold must be >= 1")


@dataclass
class PositionResult:
    position: np.ndarray  # -1 / 0 / +1 at each close
    exit_reason: np.ndarray  # code per bar (nonzero on the bar a position is closed)
    entry_z: np.ndarray  # z at entry on the entry bar, NaN elsewhere


def generate_positions(z, params: PairParams, trade_start: int = 0) -> PositionResult:
    zv = np.asarray(z, dtype=float)
    n = len(zv)
    pos = np.zeros(n, dtype=int)
    reason = np.zeros(n, dtype=int)
    entry_z = np.full(n, np.nan)
    state, held, blocked = 0, 0, 0
    for t in range(n):
        zt = zv[t]
        if t < trade_start:
            continue
        if np.isnan(zt):
            if state != 0:
                reason[t], state = 4, 0
            pos[t] = state
            continue
        if blocked != 0 and abs(zt) <= params.exit:
            blocked = 0
        if state == 0:
            if params.entry < abs(zt) < params.stop:
                direction = -1 if zt > 0 else 1
                if direction != blocked:
                    state, held = direction, 0
                    entry_z[t] = zt
        else:
            held += 1
            u = -state * zt
            if u >= params.stop:
                reason[t], blocked, state = 2, state, 0
            elif u <= params.exit:
                reason[t], state = 1, 0
            elif held >= params.max_hold:
                reason[t], blocked, state = 3, state, 0
        pos[t] = state
    return PositionResult(pos, reason, entry_z)


def trades_table(
    index: pd.DatetimeIndex, res: PositionResult, pnl: pd.Series | None = None
) -> pd.DataFrame:
    """One row per trade.

    A trade is a run of identical nonzero positions starting at bar ``i`` (the entry close).  It
    ends at the first later bar ``e`` where the position is 0 (the exit close, whose reason is
    recorded), or is still ``"open"`` at the end of the series.  The position held from the close of
    ``t`` earns the return of bar ``t + 1``, so a trade's P&L is the sum of ``pnl`` over bars
    ``i + 1 .. e``.
    """
    pos = res.position
    n = len(pos)
    rows, i = [], 0
    while i < n:
        if pos[i] == 0 or (i > 0 and pos[i - 1] == pos[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and pos[j + 1] == pos[i]:
            j += 1
        exit_bar = j + 1 if j + 1 < n else None  # pos[exit_bar] == 0 by the state machine
        last = exit_bar if exit_bar is not None else n - 1
        rows.append(
            {
                "entry": index[i],
                "exit": index[exit_bar] if exit_bar is not None else pd.NaT,
                "direction": int(pos[i]),
                "bars_held": last - i,
                "exit_reason": REASONS[int(res.exit_reason[exit_bar])]
                if exit_bar is not None
                else "open",
                "entry_z": float(res.entry_z[i]),
                "pnl": float(pnl.iloc[i + 1 : last + 1].sum()) if pnl is not None else np.nan,
            }
        )
        i = j + 1
    return pd.DataFrame(
        rows, columns=["entry", "exit", "direction", "bars_held", "exit_reason", "entry_z", "pnl"]
    )
