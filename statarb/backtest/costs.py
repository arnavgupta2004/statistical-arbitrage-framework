"""Transaction-cost model for the pair strategy.

Every assumption is explicit here because costs are where backtests are most easily flattered.

What is charged, per leg, on the traded notional ``Q`` (dollars) of a bar::

    spread      Q * half_spread                       crossing the quoted spread
    commission  Q * (commission_bps + regulatory_bps) / 1e4
    impact      Q * Y * sigma * sqrt(Q / ADV)         square-root market impact
    borrow      short notional held * borrow_bps / 1e4 / 252   per day, on the short leg

The result is expressed in *pair-capital units* (the units of ``pnl``): the traded amount in the
frame is per unit of pair capital, and one unit is ``capital / slots`` dollars.

Assumptions and limits (all configurable, all reported):

* **Execution at the decision close.**  A signal at the close of ``t`` trades at that close
  (market-on-close).  No slippage beyond the components above, no partial fills, no rejections.
* **Spread.**  Free data has no historical quotes.  The *central* model is a documented tier by
  trailing dollar ADV (``TIERS``): a conservative-to-realistic reading of S&P 500 quoted
  half-spreads of roughly 1-3 bps in liquid names.  A Corwin-Schultz (2012) estimate from daily
  high/low is available as a sensitivity; it is noisy and known to be biased *upwards* for very
  liquid stocks, so it is not the central case.
* **Impact.**  ``Y * sigma * sqrt(Q/ADV)`` is charged on the whole trade (the average, not the
  marginal, concession), with ``sigma`` the trailing daily volatility.  ``Y`` is of order 0.5-1 in
  the literature and is swept.  Impact depends on the *dollar size*, hence on the ``capital``
  assumption: it is the one component that grows faster than trade size.
* **Borrow.**  Shorts pay a fee; general-collateral borrow for large caps is about 25-50 bps a year.
  Hard-to-borrow names are not modelled (S&P 500 names almost never are) and neither is financing
  of the long leg (the book is roughly dollar-neutral and cash-collateralised).
* **Causal inputs.**  ADV, volatility and spread estimates use data through ``t - 1`` (a trade
  decided at the close of ``t`` cannot know ``t``'s own volume); tests check this with the
  leakage detectors.
* Participation above ``max_participation`` of ADV is *flagged* (``breach``), not cured: the
  formula is not calibrated far outside a few per cent of ADV.  Stage 9 (capacity) studies it.

Corwin-Schultz (2012), for two consecutive days with highs ``H`` and lows ``L``::

    beta  = ln(H_t/L_t)^2 + ln(H_{t-1}/L_{t-1})^2
    gamma = ln(max(H_t, H_{t-1}) / min(L_t, L_{t-1}))^2
    alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) - sqrt(gamma / (3 - 2 sqrt 2))
    S     = 2 (e^alpha - 1) / (1 + e^alpha)

with the paper's overnight adjustment (a prior close outside today's range shifts the range).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

K_CS = 3.0 - 2.0 * np.sqrt(2.0)

# half-spread (bps) by trailing dollar ADV: (lower bound in $, bps); documented assumption
TIERS = ((5e8, 1.0), (1e8, 1.5), (2.5e7, 2.5), (0.0, 5.0))


class CostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capital: float = Field(1e8, gt=0, description="total capital in dollars")
    slots: int = Field(20, ge=1, description="must equal the strategy's slots")
    spread_model: Literal["tiered", "corwin_schultz", "max"] = "tiered"
    spread_mult: float = Field(1.0, ge=0, description="scales the half-spread")
    commission_bps: float = Field(0.5, ge=0)
    regulatory_bps: float = Field(
        0.25, ge=0, description="SEC/FINRA fees, charged on all traded notional"
    )
    borrow_bps_per_year: float = Field(50.0, ge=0)
    impact_y: float = Field(0.5, ge=0, description="Y in Y * sigma * sqrt(Q / ADV)")
    adv_window: int = Field(20, ge=5)
    vol_window: int = Field(20, ge=5)
    cs_window: int = Field(60, ge=20)
    max_participation: float = Field(0.10, gt=0)


def corwin_schultz(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Raw daily Corwin-Schultz spread estimate (a fraction of price); can be negative.

    ``close`` enables the overnight adjustment.  Ratios only, so the price basis is irrelevant.
    """
    h, lo = high.copy(), low.copy()
    if close is not None:
        prev = close.shift(1)
        gap_down = prev - h  # prior close above today's whole range: the day gapped down
        gap_up = lo - prev  # prior close below today's whole range: the day gapped up
        # remove the gap: the range as if the day had opened at the prior close
        gd = gap_down.where(gap_down > 0, 0.0).fillna(0.0)
        gu = gap_up.where(gap_up > 0, 0.0).fillna(0.0)
        h, lo = h + (gd - gu), lo + (gd - gu)
    with np.errstate(invalid="ignore", divide="ignore"):  # bad bars (h < l) give NaN, not a warning
        beta_day = np.log(h / lo) ** 2
        beta = beta_day + beta_day.shift(1)
        gamma = np.log(np.maximum(h, h.shift(1)) / np.minimum(lo, lo.shift(1))) ** 2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / K_CS - np.sqrt(gamma / K_CS)
        return 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))


def tier_half_spread(adv: pd.DataFrame) -> pd.DataFrame:
    """Half-spread (fraction of price) from trailing dollar ADV via ``TIERS``."""
    out = pd.DataFrame(np.nan, index=adv.index, columns=adv.columns)
    for lower, bps in reversed(TIERS):
        out = out.where(~(adv >= lower), bps / 1e4)
    return out.where(adv.notna())


@dataclass
class MarketData:
    """Causal per-ticker daily inputs to the cost model (each value uses data through ``t - 1``)."""

    adv: pd.DataFrame  # trailing median dollar volume
    sigma: pd.DataFrame  # trailing daily return volatility
    half_spread: pd.DataFrame  # fraction of price


def build_market_data(
    dollar_volume: pd.DataFrame,
    ret: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame | None,
    cfg: CostConfig,
) -> MarketData:
    adv = dollar_volume.rolling(cfg.adv_window, min_periods=10).median().shift(1)
    sigma = ret.rolling(cfg.vol_window, min_periods=10).std().shift(1)
    tiered = tier_half_spread(adv)
    if cfg.spread_model == "tiered":
        half = tiered
    else:
        cs = (
            corwin_schultz(high, low, close).rolling(cfg.cs_window, min_periods=20).mean().shift(1)
            / 2
        )
        cs = cs.clip(lower=0.0)
        half = cs if cfg.spread_model == "corwin_schultz" else np.maximum(cs, tiered)
    return MarketData(adv, sigma, half)


class PairCostModel:
    """``cost_fn(frame, y, x)`` for ``run_pairs``: per-day cost components in pair-capital units."""

    def __init__(self, market: MarketData, cfg: CostConfig):
        self.market, self.cfg = market, cfg
        self.unit_dollars = cfg.capital / cfg.slots

    def _leg(self, ticker: str, trade: np.ndarray, index: pd.DatetimeIndex):
        q = trade * self.unit_dollars  # dollars traded
        adv = self.market.adv[ticker].reindex(index).to_numpy()
        sig = self.market.sigma[ticker].reindex(index).to_numpy()
        hs = self.market.half_spread[ticker].reindex(index).to_numpy()
        traded = trade > 0
        if np.any(traded & (np.isnan(adv) | np.isnan(sig) | np.isnan(hs))):
            raise ValueError(
                f"{ticker}: a trade occurs where ADV, volatility or spread is undefined"
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            participation = np.where(traded, q / adv, 0.0)
        spread = trade * np.nan_to_num(hs) * self.cfg.spread_mult
        impact = (
            trade * self.cfg.impact_y * np.nan_to_num(sig) * np.sqrt(np.nan_to_num(participation))
        )
        return spread, impact, participation

    def __call__(self, frame: pd.DataFrame, y: str, x: str) -> pd.DataFrame:
        idx = frame.index
        sp_y, im_y, part_y = self._leg(y, frame["trade_y"].to_numpy(), idx)
        sp_x, im_x, part_x = self._leg(x, frame["trade_x"].to_numpy(), idx)
        commission = (
            frame["trade"].to_numpy() * (self.cfg.commission_bps + self.cfg.regulatory_bps) / 1e4
        )
        held_y, held_x = np.roll(frame["w_y"].to_numpy(), 1), np.roll(frame["w_x"].to_numpy(), 1)
        held_y[0] = held_x[0] = 0.0
        short = np.where(held_y < 0, -held_y, 0.0) + np.where(held_x < 0, -held_x, 0.0)
        borrow = short * self.cfg.borrow_bps_per_year / 1e4 / 252.0
        out = pd.DataFrame(
            {
                "cost_spread": sp_y + sp_x,
                "cost_commission": commission,
                "cost_impact": im_y + im_x,
                "cost_borrow": borrow,
                "participation": np.maximum(part_y, part_x),
            },
            index=idx,
        )
        out["cost"] = out[["cost_spread", "cost_commission", "cost_impact", "cost_borrow"]].sum(
            axis=1
        )
        out["breach"] = out["participation"] > self.cfg.max_participation
        return out


def breakeven_multiple(gross_mean: float, cost_mean: float) -> float:
    """Factor by which *all* costs could be scaled before net reaches zero (0 if gross <= 0)."""
    if gross_mean <= 0:
        return 0.0
    return float(gross_mean / cost_mean) if cost_mean > 0 else float("inf")


class BookCostModel:
    """The same cost components for a (D, N) weight book; weights are fractions of *total* capital.

    ``trade`` and ``held`` are (D, N) arrays in capital units (``held`` is the weight carried
    through each bar, used for borrow); ``adv``, ``sigma`` and ``half_spread`` are the matching
    causal arrays from ``MarketData``.  Returns per-day components summed over names, in capital
    units -- directly comparable to the pair portfolio's return on capital.
    """

    def __init__(self, cfg: CostConfig):
        self.cfg = cfg

    def __call__(
        self,
        trade: np.ndarray,
        held: np.ndarray,
        adv: np.ndarray,
        sigma: np.ndarray,
        half_spread: np.ndarray,
    ) -> dict[str, np.ndarray]:
        traded = trade > 0
        if np.any(traded & ~(np.isfinite(adv) & np.isfinite(sigma) & np.isfinite(half_spread))):
            raise ValueError("a trade occurs where ADV, volatility or spread is undefined")
        with np.errstate(invalid="ignore", divide="ignore"):
            part = np.where(traded, trade * self.cfg.capital / adv, 0.0)
        spread = trade * np.nan_to_num(half_spread) * self.cfg.spread_mult
        impact = trade * self.cfg.impact_y * np.nan_to_num(sigma) * np.sqrt(np.nan_to_num(part))
        commission = trade * (self.cfg.commission_bps + self.cfg.regulatory_bps) / 1e4
        borrow = np.where(held < 0, -held, 0.0) * self.cfg.borrow_bps_per_year / 1e4 / 252.0
        out = {
            "cost_spread": spread.sum(1),
            "cost_commission": commission.sum(1),
            "cost_impact": impact.sum(1),
            "cost_borrow": borrow.sum(1),
            "participation": part.max(1),
        }
        out["cost"] = (
            out["cost_spread"] + out["cost_commission"] + out["cost_impact"] + out["cost_borrow"]
        )
        return out
