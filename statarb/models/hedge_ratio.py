"""Hedge-ratio estimators for ``ly = alpha + beta * lx`` -- all strictly causal.

======================  =====================================================================
``static``              OLS on the training window only, then frozen (the spec's baseline)
``expanding``           OLS re-estimated each day on *all* data up to that day
``rolling``             OLS over the trailing ``window`` days
``kalman``              random-walk state-space filter (``models/kalman.py``): a stretch goal
======================  =====================================================================

At index ``t`` every method uses observations ``<= t`` only (``static`` uses ``<= train_end``), so
the path can be computed once over a long series and consumed by a backtest without look-ahead;
the tests verify this with the Stage-1 leakage detectors.  No method is assumed better than
another: a static hedge has the least estimation noise and the most exposure to drift, a short
window the reverse, and the tests and experiments measure that trade-off rather than assert it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.models.kalman import kalman_hedge
from statarb.models.rolling import rolling_beta

METHODS = ("static", "expanding", "rolling", "kalman")


@dataclass(frozen=True)
class HedgeSpec:
    method: str = "static"
    window: int | None = None  # rolling: trailing days
    delta: float = 1e-4  # kalman: state noise per day
    min_periods: int = 60  # expanding / kalman warm-up

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.method == "rolling" and (self.window is None or self.window < 20):
            raise ValueError("rolling needs window >= 20")

    @property
    def label(self) -> str:
        if self.method == "rolling":
            return f"rolling{self.window}"
        if self.method == "kalman":
            return f"kalman{self.delta:g}"
        return self.method


def static_fit(y, x) -> tuple[float, float]:
    """OLS ``(alpha, beta)`` of ``y`` on ``x``."""
    yv, xv = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    beta, alpha = np.polyfit(xv, yv, 1)
    return float(alpha), float(beta)


def hedge_path(ly: pd.Series, lx: pd.Series, spec: HedgeSpec, train_end) -> pd.DataFrame:
    """Causal ``alpha_t``, ``beta_t`` for every date in ``ly.index`` (NaN during warm-up)."""
    if spec.method == "static":
        end = pd.Timestamp(train_end)
        alpha, beta = static_fit(ly.loc[:end], lx.loc[:end])
        return pd.DataFrame({"alpha": alpha, "beta": beta}, index=ly.index)
    if spec.method == "expanding":
        beta, alpha = rolling_beta(ly, lx, None, min_periods=spec.min_periods)
    elif spec.method == "rolling":
        beta, alpha = rolling_beta(ly, lx, spec.window, min_periods=spec.window)
    else:
        k = kalman_hedge(ly, lx, spec.delta, spec.min_periods)
        return k[["alpha", "beta"]]
    return pd.DataFrame({"alpha": alpha, "beta": beta})
