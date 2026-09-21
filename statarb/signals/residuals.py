"""Residual signals from a rolling PCA factor model (Avellaneda & Lee, 2010).

Causality contract.  Everything used to score a stock on decision day ``t`` (the close of ``t``) is
known at that close, and the *parameters* are fitted on data strictly before ``t``::

    PCA fit    rows [s - fit_window, s - 1] for a segment starting at s (refit every
               ``refit_every`` days).  The factor space is fixed inside the segment, so a stock's
               rolling betas never straddle a rotated factor definition (factor returns are
               recomputed with the current fit's weights over the whole beta window).
    betas      no-intercept OLS of the stock's returns on the k factor returns, rows [t - W, t - 1]
    residuals  in-window residuals E for rows [t - W, t - 1], and the OUT-OF-SAMPLE residual of day
               t, eps_t = r_t - beta' F_t, computed with betas that never saw day t

Two scores, both "standardised distance of the stock from its equilibrium; positive = rich (short)":

``sscore``   (Avellaneda-Lee) the cumulative residual X_j = sum of eps through j is modelled as an
             OU process: AR(1) ``X_{j+1} = a + b X_j + xi`` on the in-window path,
             ``kappa = -ln(b) * 252``, ``m = a / (1 - b)``, ``sigma_eq = sqrt(var(xi) / (1 - b^2))``
             and ``s = (X_t - m) / sigma_eq``.  Stocks with ``b`` outside (0, 1) or
             ``kappa <= kappa_min`` (mean reversion slower than ~30 days) get no score.  (Not used:
             A-L's cross-sectional demeaning of ``m``, and a Kendall bias correction of ``b``.)
``reversal`` short-horizon residual reversal (Lehmann 1990; Lo & MacKinlay 1990; Khandani & Lo
             2011): ``s = sum of the last h residuals / (sigma_E sqrt(h))`` with ``sigma_E`` the
             in-window residual volatility.  Theory: compensation for providing liquidity to
             order-flow imbalance, which mean-reverts within days.

``n_factors = 0`` is the *no-factor-model control*: residual = raw return.

Missing data: a stock with any NaN in its regression window (or on ``t``) is not scored; NaNs in the
fit window drop the stock from the PCA fit only; a missing return in factor construction is treated
as a zero standardised return.  ``score`` is NaN whenever anything is unavailable, and the state
machine flattens on NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from statarb.models.pca import fit_pca

TRADING_DAYS = 252.0


class ResidualConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_factors: int = Field(10, ge=0, description="0 = no factor model (raw returns)")
    fit_window: int = Field(504, ge=60, description="days of returns the PCA is fitted on")
    refit_every: int = Field(21, ge=1)
    beta_window: int = Field(60, ge=20, description="regression / OU window")
    signal: Literal["sscore", "reversal"] = "sscore"
    reversal_days: int = Field(5, ge=1)
    kappa_min: float = Field(8.4, gt=0, description="252/30: mean-reversion faster than ~30 days")


@dataclass
class ResidualSignals:
    dates: pd.DatetimeIndex  # decision dates
    tickers: list[str]
    score: np.ndarray  # (D, N) NaN = no score
    sigma_e: np.ndarray  # (D, N) residual daily volatility from data through t - 1
    beta: np.ndarray  # (D, N, k) factor loadings (per unit of factor return)
    eps: np.ndarray  # (D, N) the out-of-sample residual return of day t
    fits: list[dict] = field(default_factory=list)


def _ou_score(x: np.ndarray, w: int, kappa_min: float) -> np.ndarray:
    """A-L s-score from the cumulative residual path ``x`` (w + 1, N); row w is day ``t``."""
    x0, x1 = x[: w - 1], x[1:w]
    xm, ym = x0.mean(0), x1.mean(0)
    dx = x0 - xm
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (dx * (x1 - ym)).sum(0) / (dx**2).sum(0)
        a = ym - b * xm
        s2 = ((x1 - a - b * x0) ** 2).sum(0) / (w - 3)
        good = (b > 0) & (-np.log(b) * TRADING_DAYS > kappa_min)
        m = a / (1 - b)
        s_eq = np.sqrt(s2 / (1 - b**2))
        s = (x[w] - m) / s_eq
    return np.where(good & np.isfinite(s), s, np.nan)


def residual_signals(returns: pd.DataFrame, first: int, cfg: ResidualConfig) -> ResidualSignals:
    """Score every row ``first .. len-1`` of ``returns`` (dates x tickers, NaN = missing)."""
    r_all = returns.to_numpy(dtype=float)
    n_rows, n = r_all.shape
    k, w, seg_len = cfg.n_factors, cfg.beta_window, cfg.refit_every
    if first < cfg.fit_window:
        raise ValueError(f"need {cfg.fit_window} rows of history before the first decision")
    d_total = n_rows - first
    score = np.full((d_total, n), np.nan)
    sigma_e = np.full((d_total, n), np.nan)
    beta_out = np.full((d_total, n, k), np.nan)
    eps_out = np.full((d_total, n), np.nan)
    fits = []
    for s in range(first, n_rows, seg_len):
        e = min(s + seg_len, n_rows)
        base = s - w
        if k > 0:
            block = r_all[s - cfg.fit_window : s]
            ok = np.isfinite(block).all(axis=0)
            fit = fit_pca(block[:, ok])
            z = np.nan_to_num((r_all[base:e][:, ok] - fit.mean) / fit.scale)
            f_all = z @ fit.components[:, :k]
            fits.append(
                {
                    "date": returns.index[s],
                    "n_fit": int(ok.sum()),
                    "explained": fit.explained_ratio(k),
                    "mp_k": fit.marchenko_pastur_k(),
                }
            )
        else:
            f_all = np.zeros((e - base, 0))
        for t in range(s, e):
            rw, rt = r_all[t - w : t], r_all[t]
            valid = np.isfinite(rw).all(axis=0) & np.isfinite(rt)
            rw0, rt0 = np.nan_to_num(rw), np.nan_to_num(rt)
            fw, ft = f_all[t - w - base : t - base], f_all[t - base]
            if k > 0:
                b = np.linalg.lstsq(fw, rw0, rcond=None)[0]
                res_w, eps_t = rw0 - fw @ b, rt0 - ft @ b
            else:
                b = np.zeros((0, n))
                res_w, eps_t = rw0, rt0
            sig = np.sqrt((res_w**2).sum(0) / (w - k))
            if cfg.signal == "sscore":
                sc = _ou_score(np.cumsum(np.vstack([res_w, eps_t]), axis=0), w, cfg.kappa_min)
            else:
                h = cfg.reversal_days
                with np.errstate(invalid="ignore", divide="ignore"):
                    sc = (eps_t + res_w[w - h + 1 :].sum(0)) / (sig * np.sqrt(h))
            d = t - first
            score[d] = np.where(valid, sc, np.nan)
            sigma_e[d] = np.where(valid, sig, np.nan)
            beta_out[d] = np.where(valid[:, None], b.T, np.nan)
            eps_out[d] = np.where(valid, eps_t, np.nan)
    return ResidualSignals(
        returns.index[first:], list(returns.columns), score, sigma_e, beta_out, eps_out, fits
    )


def states_from_scores(
    score: np.ndarray,
    entry: float = 1.25,
    exit: float = 0.5,
    stop: float = 4.0,
    max_hold: int = 60,
) -> np.ndarray:
    """Per-stock position state in {-1, 0, +1} from a (D, N) score matrix; the last row is flat.

    Long when ``score < -entry`` (cheap), short when ``score > entry`` (rich).  Exit when the score
    returns inside ``+-exit``.  A stop (``|score| > stop``) or a time stop (``max_hold`` days) exits
    and blocks re-entry until the score has re-entered the exit band.  NaN flattens.  The state on
    row ``d`` is the position established at the close of ``d`` (it earns bar ``d + 1``).
    """
    d_total, n = score.shape
    out = np.zeros((d_total, n), dtype=np.int8)
    state = np.zeros(n, dtype=np.int8)
    age = np.zeros(n, dtype=np.int64)
    blocked = np.zeros(n, dtype=bool)
    for d in range(d_total):
        x = score[d]
        nan = np.isnan(x)
        xx = np.where(nan, 0.0, x)
        long_, short_ = state > 0, state < 0
        mean_exit = (long_ & (xx > -exit)) | (short_ & (xx < exit))
        stopped = (long_ & (xx < -stop)) | (short_ & (xx > stop))
        timed = (state != 0) & (age >= max_hold)
        forced = (stopped | timed) & ~nan
        state = np.where(nan | mean_exit | stopped | timed, 0, state).astype(np.int8)
        blocked = (blocked | forced) & ~((xx > -exit) & (xx < exit))
        free = (state == 0) & ~nan & ~blocked
        state = np.where(
            free & (xx < -entry) & (xx > -stop),
            1,
            np.where(free & (xx > entry) & (xx < stop), -1, state),
        ).astype(np.int8)
        age = np.where(state != 0, age + 1, 0)
        out[d] = state
    out[-1] = 0
    return out
