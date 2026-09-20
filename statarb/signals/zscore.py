"""Spread z-score with a (possibly time-varying) hedge ratio, in closed form.

With hedge ratio ``b`` the spread over a trailing window is ``S_s = ly_s - b * lx_s``.  Its window
mean and variance follow from the window moments of ``ly`` and ``lx``::

    mean_S = m_y - b m_x            var_S = v_y - 2 b c_xy + b^2 v_x

and the z-score at ``t`` is ``(S_t - mean_S) / sd_S`` -- the intercept cancels.

The point of computing it this way: for a time-varying ``b_t`` the *current* hedge ratio is applied
to the whole window, so a change in ``b`` moves the level of the spread consistently and does not
create a spurious jump (a spread stitched together from each day's own ``b_s`` would jump by
``delta_b * lx``, with ``lx`` near 4).  Every hedge method (static, expanding, rolling, Kalman) goes
through this one construction, so comparisons isolate the hedge ratio itself.  All moments are
trailing and include day ``t``: information available at the close of ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _moments(x: pd.Series, window: int | None, min_periods: int | None):
    return (
        x.expanding(min_periods=min_periods or 20)
        if window is None
        else x.rolling(window, min_periods=min_periods or window)
    )


def spread_zscore(
    ly: pd.Series,
    lx: pd.Series,
    beta: pd.Series,
    window: int | None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Columns ``z``, ``sd`` (window sd of the spread under today's ``beta``), ``spread``."""
    my, mx = _moments(ly, window, min_periods).mean(), _moments(lx, window, min_periods).mean()
    vy, vx = _moments(ly, window, min_periods).var(), _moments(lx, window, min_periods).var()
    cxy = _moments(ly, window, min_periods).cov(lx)
    var_s = vy - 2.0 * beta * cxy + beta**2 * vx
    sd = np.sqrt(var_s.where(var_s > 1e-14))
    z = ((ly - my) - beta * (lx - mx)) / sd
    return pd.DataFrame({"z": z, "sd": sd, "spread": ly - beta * lx})
