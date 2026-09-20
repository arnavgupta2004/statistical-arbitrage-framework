"""Pair discovery: a hierarchical, training-window-only screen with an empirical null.

Funnel (every step sees the training window only; nothing after ``train_end`` is read)::

    point-in-time members on train_end            (universe, identity-verified)
      -> eligibility (coverage, liquidity, stale prices, non-ordinary dividends, sector known)
      -> candidate pairs: each stock's k best-correlated peers inside its sector (or cluster)
      -> Engle-Granger, BOTH regression directions; statistic T = min of the two
      -> diagnostics: OU half-life, spread volatility, hedge-ratio and stationarity stability
      -> calibrated p-value, fixed a-priori selection rule, ranking by an economic proxy

Why an empirical null
---------------------
Stage 2 showed the Engle-Granger p-value is not an error rate on real prices (5-year windows reject
7-15 % of independent pairs at a nominal 5 %), and taking the better of two directions inflates it
further.  So the p-value used here is *empirical*: the **whole pipeline** (correlation neighbours ->
both-direction test) is re-run on ``n_null_panels`` panels in which every stock's returns are
circularly shifted by an independent random offset.  That keeps each stock's fat tails, volatility
clustering and autocorrelation, destroys all cross-stock dependence (so no pair can be
cointegrated), and applies the *same selection* (top-k correlated peers, best of two directions) to
the null.  The calibrated p-value of a real candidate is the share of null candidates with a
statistic at least as extreme.  The same null gives a permutation estimate of the false-discovery
proportion among the screen's discoveries.

What this does and does not buy
-------------------------------
* It calibrates the *per-candidate* p-value against real-return dynamics and against the selection
  effect of the neighbour filter.  It does **not** correct for the number of candidates: with N
  candidates, ``alpha * N`` false discoveries are expected at level alpha.  ``estimated_fdr``
  reports exactly that; the family-wise / FDR corrections proper are Stage 8, and the counts they
  need (``n_family``, ``n_candidates``, ``n_tests``) are recorded.
* Independent circular shifts leave a small fraction of stock pairs with nearly equal shifts (hence
  aligned volatility regimes); with hundreds of stocks this is unavoidable and is a known
  imperfection.
* The null has no factor structure (the market factor is destroyed).  ``tests/test_screening.py``
  checks the calibration on a factor-null panel (common market factor, idiosyncratic random walks).
* Sectors are today's GICS labels (look-ahead in classification); the selection rule's thresholds
  are fixed in ``ScreenConfig`` before any result is seen and must not be tuned on the screen's
  output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from statarb.models.ou import fit_ou
from statarb.selection.adf import adf_test
from statarb.selection.clustering import cluster_groups
from statarb.selection.cointegration import engle_granger
from statarb.selection.correlation import n_group_pairs, return_correlation, top_k_neighbours
from statarb.selection.hurst import hurst_variance

if TYPE_CHECKING:
    from statarb.data.cleaning.alignment import AlignedPanel
    from statarb.data.pipeline import DataPipeline


class ScreenConfig(BaseModel):
    """Every knob of the screen.  Defaults are the a-priori specification (not tuned)."""

    model_config = ConfigDict(extra="forbid")

    # candidate generation
    candidate_method: Literal["correlation", "cluster"] = "correlation"
    k_neighbours: int = Field(5, ge=1)
    same_sector: bool = True
    cluster_target_size: int = Field(8, ge=2)
    # cointegration test
    trend: Literal["c", "ct"] = "c"
    maxlag: int = Field(
        10, ge=0, description="ADF augmentation cap (speed; residuals are near-white)"
    )
    autolag: Literal["aic", "bic"] = "aic"
    # empirical null
    n_null_panels: int = Field(10, ge=1)
    seed: int = 0
    # a-priori selection rule (economically motivated, NOT tuned on results)
    alpha: float = Field(0.05, gt=0, lt=1, description="calibrated-p threshold")
    half_life_min: float = Field(5.0, gt=0, description="trading days; faster is untradeable")
    half_life_max: float = Field(60.0, gt=0, description="trading days; slower ties up capital")
    require_positive_beta: bool = True
    max_pairs: int = Field(20, ge=1)
    # eligibility (causal: computed inside the training window)
    min_coverage: float = Field(1.0, gt=0, le=1)
    min_median_dollar_volume: float = 5e6
    max_zero_volume_share: float = Field(0.01, ge=0, le=1)
    max_dividend_yield: float = Field(0.10, gt=0)


# ---------------------------------------------------------------------------------------------
# statistics of one pair
# ---------------------------------------------------------------------------------------------
def _both_directions(a: np.ndarray, b: np.ndarray, cfg: ScreenConfig):
    fwd = engle_granger(a, b, trend=cfg.trend, maxlag=cfg.maxlag, autolag=cfg.autolag)
    rev = engle_granger(b, a, trend=cfg.trend, maxlag=cfg.maxlag, autolag=cfg.autolag)
    return fwd, rev


def _statistic_only(a: np.ndarray, b: np.ndarray, cfg: ScreenConfig) -> float:
    """``T = min(stat_ab, stat_ba)``: the statistic whose null the calibration reproduces."""
    fwd, rev = _both_directions(a, b, cfg)
    return min(fwd.statistic, rev.statistic)


def _diagnostics(spread: np.ndarray, y: np.ndarray, x: np.ndarray, cfg: ScreenConfig) -> dict:
    """Mean-reversion, stability and economic-proxy diagnostics for one training-window spread."""
    ou = fit_ou(spread, bias_correct=True)
    n = len(spread)
    h = n // 2
    out = {
        "kappa": ou.kappa,
        "half_life": ou.half_life,
        "half_life_lo": ou.half_life_ci[0],
        "half_life_hi": ou.half_life_ci[1],
        "spread_sd": float(np.std(spread, ddof=1)),
        "stationary_sd": ou.stationary_sd,
        # secondary, exploratory only (see selection/hurst.py); never used to select
        "hurst": hurst_variance(spread, max_lag=min(20, n // 8)) if n >= 160 else float("nan"),
        # cheap economic proxy: reversion speed x amplitude, in bps of log-price per day
        "edge_bps_per_day": 1e4 * ou.kappa * ou.stationary_sd
        if ou.mean_reverting
        else float("nan"),
    }
    b1 = np.polyfit(x[:h], y[:h], 1)[0]
    b2 = np.polyfit(x[h:], y[h:], 1)[0]
    beta_full = np.polyfit(x, y, 1)[0]
    out["beta_half1"], out["beta_half2"] = float(b1), float(b2)
    out["beta_rel_change"] = float(abs(b2 - b1) / abs(beta_full)) if beta_full else float("nan")
    for name, seg in (("adf_p_half1", spread[:h]), ("adf_p_half2", spread[h:])):
        try:
            out[name] = adf_test(seg, "c", maxlag=cfg.maxlag, autolag=cfg.autolag).pvalue
        except ValueError:
            out[name] = float("nan")
    return out


# ---------------------------------------------------------------------------------------------
# null panels
# ---------------------------------------------------------------------------------------------
def shift_panel(logp: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Independent-by-construction panel: each stock's returns circularly shifted by its own offset.

    Marginal dynamics (fat tails, volatility clustering, autocorrelation) are preserved exactly; the
    multiset of returns of every stock is unchanged; every cross-stock dependence is destroyed.
    """
    r = logp.diff().iloc[1:].to_numpy()
    n = len(r)
    shifts = rng.integers(1, n, size=r.shape[1])
    shifted = np.column_stack([np.roll(r[:, j], int(s)) for j, s in enumerate(shifts)])
    levels = (
        np.vstack([np.zeros((1, r.shape[1])), np.cumsum(shifted, axis=0)]) + logp.iloc[0].to_numpy()
    )
    return pd.DataFrame(levels, index=logp.index, columns=logp.columns)


def _groups(corr: pd.DataFrame, sectors: pd.Series, cfg: ScreenConfig) -> pd.Series | None:
    if cfg.candidate_method == "cluster":
        return cluster_groups(corr, sectors, cfg.cluster_target_size)
    return sectors.reindex(corr.columns) if cfg.same_sector else None


def _candidates(logp: pd.DataFrame, sectors: pd.Series, cfg: ScreenConfig):
    corr = return_correlation(logp)
    return corr, top_k_neighbours(corr, _groups(corr, sectors, cfg), cfg.k_neighbours)


def null_statistics(
    logp: pd.DataFrame, sectors: pd.Series, cfg: ScreenConfig, rng
) -> list[np.ndarray]:
    """Run the candidate-selection + test pipeline on ``n_null_panels`` independent panels."""
    out = []
    for _ in range(cfg.n_null_panels):
        panel = shift_panel(logp, rng)
        _, pairs = _candidates(panel, sectors, cfg)
        arr = panel.to_numpy()
        col = {c: i for i, c in enumerate(panel.columns)}
        stats = []
        for a, b in pairs:
            try:
                stats.append(_statistic_only(arr[:, col[a]], arr[:, col[b]], cfg))
            except ValueError:
                continue
        out.append(np.asarray(stats))
    return out


def empirical_pvalue(t_obs, null_pool: np.ndarray) -> np.ndarray:
    """``(1 + #{null <= t}) / (1 + B)``: never zero, valid for a lower-tail statistic."""
    null_sorted = np.sort(null_pool)
    k = np.searchsorted(null_sorted, np.asarray(t_obs, dtype=float), side="right")
    return (1.0 + k) / (1.0 + len(null_sorted))


# ---------------------------------------------------------------------------------------------
# the screen
# ---------------------------------------------------------------------------------------------
@dataclass
class ScreenResult:
    pairs: pd.DataFrame  # one row per tested candidate, sorted by calibrated p then edge
    config: dict
    n_eligible: int
    n_family: int  # all same-group pairs an exhaustive screen would have tested
    n_candidates: int  # pairs actually tested
    n_tests: int  # EG regressions run (2 per candidate)
    n_failed: int
    null_stats: list[np.ndarray] = field(repr=False, default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    train_window: tuple[str, str] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def selected(self) -> pd.DataFrame:
        return self.pairs[self.pairs["selected"]].sort_values("rank")

    def discoveries(self, alpha: float | None = None) -> int:
        a = alpha if alpha is not None else self.config["alpha"]
        return int((self.pairs["p_calibrated"] <= a).sum())

    def estimated_fdr(self, alpha: float) -> float:
        """Expected false discoveries under the null / observed discoveries at level ``alpha``.

        The null rejects ``alpha`` of its candidates by construction, so the expected number of
        false discoveries among ``n_candidates`` is ``alpha * n_candidates`` (a conservative
        pi0 = 1).
        """
        n = self.discoveries(alpha)
        return float(min(1.0, alpha * self.n_candidates / n)) if n else float("nan")

    def summary(self, alphas=(0.001, 0.005, 0.01, 0.05)) -> dict:
        p = self.pairs
        return {
            "n_eligible": self.n_eligible,
            "n_family": self.n_family,
            "n_candidates": self.n_candidates,
            "n_tests": self.n_tests,
            "n_failed": self.n_failed,
            "naive_discoveries_at_5pct": int((p["p_nominal_best"] < 0.05).sum()),
            "discoveries": {str(a): self.discoveries(a) for a in alphas},
            "expected_false": {str(a): a * self.n_candidates for a in alphas},
            "estimated_fdr": {str(a): self.estimated_fdr(a) for a in alphas},
            "n_selected": int(p["selected"].sum()),
            "n_null_candidates": [int(len(s)) for s in self.null_stats],
        }


def screen_pairs(logp: pd.DataFrame, sectors: pd.Series, cfg: ScreenConfig) -> ScreenResult:
    """Screen a gap-free panel of dividend-adjusted log prices (training window only)."""
    if logp.isna().any().any():
        raise ValueError("log prices contain NaN; eligibility must drop incomplete tickers first")
    rng = np.random.default_rng(cfg.seed)
    sectors = sectors.reindex(logp.columns)
    corr, pairs = _candidates(logp, sectors, cfg)
    arr = logp.to_numpy()
    col = {c: i for i, c in enumerate(logp.columns)}

    rows, failed = [], 0
    for a, b in pairs:
        ia, ib = col[a], col[b]
        try:
            fwd, rev = _both_directions(arr[:, ia], arr[:, ib], cfg)
        except ValueError:
            failed += 1
            continue
        use_fwd = fwd.statistic <= rev.statistic
        res, (yname, xname), (yv, xv) = (
            (fwd, (a, b), (arr[:, ia], arr[:, ib]))
            if use_fwd
            else (rev, (b, a), (arr[:, ib], arr[:, ia]))
        )
        spread = np.asarray(res.spread)
        row = {
            "y": yname,
            "x": xname,
            "sector": sectors.get(a) if sectors.get(a) == sectors.get(b) else "mixed",
            "corr": float(corr.at[a, b]),
            "T": float(min(fwd.statistic, rev.statistic)),
            "stat_ab": float(fwd.statistic),
            "stat_ba": float(rev.statistic),
            "p_nominal_best": float(min(fwd.pvalue, rev.pvalue)),
            "beta": float(res.hedge_ratio[0]),
            "alpha_intercept": float(res.intercept),
            "n_obs": int(res.nobs),
        }
        row.update(_diagnostics(spread, yv, xv, cfg))
        rows.append(row)

    df = pd.DataFrame(rows)
    null = null_statistics(logp, sectors, cfg, rng)
    pool = np.concatenate([s for s in null if len(s)]) if any(len(s) for s in null) else np.empty(0)
    if len(df) and len(pool):
        df["p_calibrated"] = empirical_pvalue(df["T"].to_numpy(), pool)
    else:
        df["p_calibrated"] = np.nan

    if len(df):
        ok = (
            (df["p_calibrated"] <= cfg.alpha)
            & df["half_life"].between(cfg.half_life_min, cfg.half_life_max)
            & ((df["beta"] > 0) if cfg.require_positive_beta else True)
        )
        df["selected"] = ok
        df = df.sort_values(["p_calibrated", "T"]).reset_index(drop=True)
        df["rank"] = np.nan
        chosen = (
            df[df["selected"]].sort_values("edge_bps_per_day", ascending=False).head(cfg.max_pairs)
        )
        df.loc[chosen.index, "rank"] = np.arange(1, len(chosen) + 1)
        df["selected"] = df["rank"].notna()
    else:
        df["selected"] = pd.Series(dtype=bool)
        df["rank"] = pd.Series(dtype=float)

    n_family = (
        n_group_pairs(sectors)
        if cfg.same_sector
        else len(logp.columns) * (len(logp.columns) - 1) // 2
    )
    return ScreenResult(
        pairs=df,
        config=cfg.model_dump(),
        n_eligible=logp.shape[1],
        n_family=n_family,
        n_candidates=len(df),
        n_tests=2 * len(df),
        n_failed=failed,
        null_stats=null,
    )


# ---------------------------------------------------------------------------------------------
# eligibility and the pipeline entry point
# ---------------------------------------------------------------------------------------------
def eligible_tickers(
    pipe: DataPipeline, train_start, train_end, cfg: ScreenConfig
) -> tuple[AlignedPanel, list[str], dict[str, str]]:
    """Point-in-time, identity-verified, liquid, gap-free tickers as of ``train_end``.

    The first failing reason is recorded for every exclusion.  Reads only data up to ``train_end``.
    """
    universe = pipe.universe()
    members = universe.members_on(pd.Timestamp(train_end))
    panel = pipe.panel(members, train_start, train_end)  # as_of = train_end: causal
    excluded: dict[str, str] = dict(panel.dropped)
    for t in panel.empty:
        excluded[t] = "no bars in the training window"

    verified = pipe.members_mask(panel, verified=True).iloc[-1]
    coverage = panel.coverage()
    zero_share = (panel.volume == 0).mean()
    dollar_vol = panel.dollar_volume.median()
    prior_close = panel.close.shift(1)
    max_yield = (panel.dividends / prior_close).max()
    sector = universe.sector_of(list(panel.close.columns))

    keep = []
    for t in panel.close.columns:
        if t in excluded:
            continue
        if not bool(verified.get(t, False)):
            excluded[t] = "price history not verified as this company's (symbol reuse / no prices)"
        elif coverage[t] < cfg.min_coverage:
            excluded[t] = f"coverage {coverage[t]:.3f} < {cfg.min_coverage}"
        elif zero_share[t] > cfg.max_zero_volume_share:
            excluded[t] = f"zero-volume share {zero_share[t]:.3f}"
        elif not dollar_vol[t] >= cfg.min_median_dollar_volume:
            excluded[t] = f"median dollar volume {dollar_vol[t]:.3g} below minimum"
        elif max_yield[t] > cfg.max_dividend_yield:
            excluded[t] = f"non-ordinary distribution ({max_yield[t]:.0%} of prior close)"
        elif cfg.same_sector and sector[t] == "Unknown":
            excluded[t] = "no sector classification"
        else:
            keep.append(t)
    return panel, keep, excluded


def discover_pairs(
    pipe: DataPipeline, train_start, train_end, cfg: ScreenConfig | None = None
) -> ScreenResult:
    """Discover candidate pairs using only information available at ``train_end``."""
    cfg = cfg or ScreenConfig()
    panel, tickers, excluded = eligible_tickers(pipe, train_start, train_end, cfg)
    if len(tickers) < 4:
        raise ValueError(f"only {len(tickers)} eligible tickers in the training window")
    logp = np.log(panel.tr_close()[tickers])  # dividend-adjusted, anchored at train_end
    sectors = pipe.universe().sector_of(tickers)
    result = screen_pairs(logp, sectors, cfg)
    result.excluded = excluded
    result.train_window = (
        str(pd.Timestamp(train_start).date()),
        str(pd.Timestamp(train_end).date()),
    )
    result.meta = {
        "n_members_train_end": len(pipe.universe().members_on(pd.Timestamp(train_end))),
        "data_fingerprint": pipe.fingerprint(),
        "universe": pipe.universe().provenance,
        "caveats": pipe.universe().caveats,
    }
    return result
