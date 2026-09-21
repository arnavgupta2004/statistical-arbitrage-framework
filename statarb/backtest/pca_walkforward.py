"""Walk-forward evaluation of the PCA residual strategy, on the same folds as the pair strategy.

Per fold (calendar-year test block, rolling training window)::

    universe   the point-in-time, identity-verified, liquid, gap-free names as of ``train_end``
               (``eligible_tickers``: the pair screen's eligibility), fixed for the block
    signals    ``residual_signals`` scores every test day from a PCA refitted every ``refit_every``
               days on the trailing ``fit_window`` returns -- never on data from the day it scores
    book       states -> inverse-residual-vol sizing -> exactly dollar/factor-neutral weights, flat
               at the block's end (positions never carry across blocks)
    P&L        the position set at the close of ``d`` earns bar ``d + 1``; turnover is drift-aware;
               costs from ``BookCostModel`` on the same causal market data as the pair strategy

Events (ex-dates of distributions above ``max_dividend_yield`` of the prior close) are handled as in
the pair engine: the event bar's return is missing (so the name is unscored while it sits in the
regression window) and the name is flat from one day before the ex-date (public calendar fact).

The **placebo** severs the link between a signal and the stock it is about while keeping everything
else: the score columns are relabelled by a random *permutation of the tradable names, the same for
every day of the block*, so each score path (its persistence, its NaN pattern, its turnover) is kept
and merely attached to another stock; sizing, neutralisation and costs then run on the stock that
actually holds the position.  Under the null "the score carries no information about that stock's
returns" the permuted book has the same construction, the same turnover and (approximately) the same
costs, with zero expected gross.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from statarb.backtest.costs import BookCostModel, CostConfig, MarketData, build_market_data
from statarb.backtest.walkforward import Fold
from statarb.portfolio.neutral import BookConfig, neutral_book
from statarb.selection.screening import ScreenConfig, eligible_tickers
from statarb.signals.residuals import (
    ResidualConfig,
    ResidualSignals,
    residual_signals,
    states_from_scores,
)


class PCAStrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    residual: ResidualConfig = ResidualConfig()
    book: BookConfig = BookConfig()
    max_dividend_yield: float = 0.10

    @property
    def label(self) -> str:
        r = self.residual
        return f"pca_{r.signal}_k{r.n_factors}"


@dataclass
class PCAFoldData:
    fold: Fold
    tickers: list[str]
    ret: pd.DataFrame  # total returns, event bars set to NaN; from train_start to test_end
    event: pd.DataFrame
    first: int  # row of test_start in ``ret``
    dollar_volume: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    n_eligible: int = 0


def prepare_pca_fold(
    pipe, fold: Fold, screen_cfg: ScreenConfig | None = None, max_dividend_yield: float = 0.10
) -> PCAFoldData:
    """Eligible universe as of ``train_end``; prices to ``test_end`` (as of ``test_end``)."""
    cfg = screen_cfg or ScreenConfig()
    _, keep, _ = eligible_tickers(pipe, fold.train_start, fold.train_end, cfg)
    panel = pipe.panel(keep, fold.train_start, fold.test_end)
    event = ((panel.dividends / panel.close.shift(1)) > max_dividend_yield).fillna(False)
    ret = panel.ret.mask(event)
    first = int(ret.index.searchsorted(fold.test_start))
    return PCAFoldData(
        fold,
        list(ret.columns),
        ret,
        event,
        first,
        panel.dollar_volume,
        panel.high,
        panel.low,
        panel.close,
        len(keep),
    )


@dataclass
class FoldSignals:
    fd: PCAFoldData
    sig: ResidualSignals
    market: dict[str, np.ndarray]  # adv, sigma, half_spread on the test rows (D, N)
    ret: np.ndarray  # (D, N) bar returns of the test rows
    tradable: np.ndarray  # (D, N) cost inputs defined
    event_next: np.ndarray  # (D, N) an event bar follows the decision day


def fold_signals(fd: PCAFoldData, cfg: PCAStrategyConfig, cost_cfg: CostConfig) -> FoldSignals:
    sig = residual_signals(fd.ret, fd.first, cfg.residual)
    md: MarketData = build_market_data(
        fd.dollar_volume, fd.ret, fd.high, fd.low, fd.close, cost_cfg
    )
    rows = slice(fd.first, len(fd.ret))
    market = {
        "adv": md.adv.iloc[rows].to_numpy(),
        "sigma": md.sigma.iloc[rows].to_numpy(),
        "half_spread": md.half_spread.iloc[rows].to_numpy(),
    }
    tradable = np.isfinite(market["adv"]) & np.isfinite(market["sigma"])
    tradable &= np.isfinite(market["half_spread"])
    ev = fd.event.to_numpy()[rows]
    nxt = np.vstack([ev[1:], np.zeros((1, ev.shape[1]), dtype=bool)])
    return FoldSignals(fd, sig, market, fd.ret.to_numpy()[rows], tradable, nxt)


def book_states(
    fs: FoldSignals, cfg: PCAStrategyConfig, perm: np.ndarray | None = None
) -> np.ndarray:
    """(D, N) position states; flat the day before an event; ``perm`` relabels columns."""
    score = np.where(fs.event_next, np.nan, fs.sig.score)
    if perm is not None:
        relabelled = np.full_like(score, np.nan)
        relabelled[:, perm] = score
        score = relabelled
    b = cfg.book
    return states_from_scores(score, b.entry, b.exit, b.stop, b.max_hold)


def book_weights(
    fs: FoldSignals,
    cfg: PCAStrategyConfig,
    perm: np.ndarray | None = None,
    state: np.ndarray | None = None,
) -> np.ndarray:
    """(D, N) weights set at each decision close.  ``perm`` relabels score columns (the placebo)."""
    if state is None:
        state = book_states(fs, cfg, perm)
    # the last row is flat: ``states_from_scores`` closes everything at the block's end
    return neutral_book(state, fs.sig.sigma_e, fs.sig.beta, fs.tradable, cfg.book)


def book_pnl(w: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P&L per bar, drift-aware trades at each close, and the weight held through each bar."""
    r0 = np.nan_to_num(r)
    held = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    pnl = (held * r0).sum(axis=1)
    trade = np.abs(w - held * (1.0 + r0))
    return pnl, trade, held


def run_fold(
    fs: FoldSignals,
    cfg: PCAStrategyConfig,
    cost_cfg: CostConfig | None = None,
    perm: np.ndarray | None = None,
) -> pd.DataFrame:
    """Daily gross / cost / net (capital units) and book diagnostics for one fold."""
    state = book_states(fs, cfg, perm)
    w = book_weights(fs, cfg, perm, state)
    pnl, trade, held = book_pnl(w, fs.ret)
    out = pd.DataFrame({"pnl": pnl}, index=fs.sig.dates)
    out["trade"] = trade.sum(axis=1)
    out["gross_exposure"] = np.abs(held).sum(axis=1)
    out["net_exposure"] = held.sum(axis=1)
    out["n_names"] = (held != 0).sum(axis=1)  # includes the hedge, which touches most names
    prev_state = np.vstack([np.zeros((1, state.shape[1])), state[:-1]])
    out["n_signal"] = (prev_state != 0).sum(axis=1)
    if cost_cfg is None:
        out["cost"] = 0.0
    else:
        m = fs.market
        comp = BookCostModel(cost_cfg)(trade, held, m["adv"], m["sigma"], m["half_spread"])
        for k, v in comp.items():
            out[k] = v
    out["net"] = out["pnl"] - out["cost"]
    return out


def walk_forward_pca(
    fold_signals_: list[FoldSignals],
    cfg: PCAStrategyConfig,
    cost_cfg: CostConfig | None = None,
    perms: dict[int, np.ndarray] | None = None,
    allowed_phases: tuple[str, ...] = ("research",),
) -> pd.DataFrame:
    """Stitched out-of-sample daily series.  ``perms`` maps fold index -> column permutation."""
    bad = {fs.fd.fold.phase for fs in fold_signals_} - set(allowed_phases)
    if bad:
        raise ValueError(f"folds in phases {sorted(bad)} are not allowed here")
    parts = []
    for fs in fold_signals_:
        p = None if perms is None else perms[fs.fd.fold.index]
        parts.append(run_fold(fs, cfg, cost_cfg, p).assign(fold=fs.fd.fold.index))
    return pd.concat(parts)


def placebo_permutation(fs: FoldSignals, rng: np.random.Generator) -> np.ndarray:
    """Random relabelling of the names that are tradable on most test days (others stay put)."""
    n = fs.tradable.shape[1]
    movable = np.flatnonzero(fs.tradable.mean(axis=0) > 0.5)
    perm = np.arange(n)
    perm[movable] = rng.permutation(movable)
    return perm
