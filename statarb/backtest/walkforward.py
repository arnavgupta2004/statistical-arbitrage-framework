"""Walk-forward evaluation: re-select pairs on a training window, trade the next block, repeat.

One fold (everything a fold reads is bounded by the dates in its definition)::

    train window [train_start, train_end]   -> point-in-time universe, eligibility, pair screen,
                                               hedge ratios (all fitted here, then frozen)
    test block   [test_start,  test_end]    -> trade the frozen pairs bar by bar; positions are
                                               forced flat at ``test_end`` so blocks are independent

Blocks are calendar years aligned with the phase boundaries (research / validation / holdout), so a
block never straddles two phases.  There is no purging or embargo because nothing is labelled with
future returns: a position opened in block ``k`` is closed at the end of block ``k``, and a fold's
selection uses only data at or before its ``train_end``.

The stitched out-of-sample return series is the concatenation of the blocks.  Portfolio construction
here is a fixed-slot equal weight: each of ``slots`` slots gets ``1/slots`` of capital and a pair's
unit-notional P&L is scaled by that (unused slots earn nothing), so the return is on total capital
and does not depend on how many pairs happened to pass the screen.

Events: ex-dates of non-ordinary distributions (spin-offs, specials; > ``max_dividend_yield`` of the
prior close) distort total returns and the trailing z-window.  Ex-dates are public ahead of time, so
a live system can stay out: the spread is blacked out from ``event_lead`` days before the ex-date to
``blackout`` days after (default: the z-window).  This is the *only* place the engine reads a
calendar fact one day ahead of a decision, and it is documented as such.

Costs: ``cost_fn(frame, y, x)`` receives a pair's frame (with per-leg ``trade_y`` / ``trade_x``)
and the two tickers, and returns a per-day cost in pair-capital units -- a Series, or a DataFrame
of components (``cost_*`` columns plus ``cost``).  The default is no costs: every result is
*gross*.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from statarb.backtest.pair_pnl import run_pair
from statarb.config import SplitConfig
from statarb.models.hedge_ratio import HedgeSpec
from statarb.selection.screening import ScreenConfig, ScreenResult, discover_pairs
from statarb.signals.pairs import PairParams


class StrategyConfig(BaseModel):
    """Everything about how a selected pair is traded (JSON-serialisable for the registry)."""

    model_config = ConfigDict(extra="forbid")

    hedge_method: Literal["static", "expanding", "rolling", "kalman"] = "static"
    hedge_window: int | None = None
    kalman_delta: float = 1e-5
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0
    max_hold: int = Field(60, ge=1)
    z_window: int = Field(60, ge=20)
    target_vol: float | None = None
    freeze_beta: bool = False
    slots: int = Field(20, ge=1, description="capital is split into this many equal pair slots")
    max_dividend_yield: float = 0.10
    event_lead: int = 1
    event_blackout: int | None = Field(None, description="days after an event; default = z_window")

    @property
    def hedge(self) -> HedgeSpec:
        return HedgeSpec(self.hedge_method, window=self.hedge_window, delta=self.kalman_delta)

    @property
    def params(self) -> PairParams:
        return PairParams(self.entry, self.exit, self.stop, self.max_hold)

    @property
    def label(self) -> str:
        return f"{self.hedge.label}_in{self.entry:g}"


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    phase: str


def make_folds(
    split: SplitConfig,
    calendar,
    first_test_year: int,
    last_test_year: int,
    train_years: int,
    mode: str = "rolling",
    data_end=None,
) -> list[Fold]:
    """Calendar-year test blocks, each trained on the ``train_years`` before it (or all history)."""
    if mode not in ("rolling", "expanding"):
        raise ValueError("mode must be 'rolling' or 'expanding'")
    folds = []
    for i, year in enumerate(range(first_test_year, last_test_year + 1)):
        ts_ = calendar.sessions(f"{year}-01-01", f"{year}-12-31")
        if data_end is not None:
            ts_ = ts_[ts_ <= pd.Timestamp(data_end)]
        if len(ts_) == 0:
            raise ValueError(f"no sessions in {year}")
        test_start, test_end = ts_[0], ts_[-1]
        lo = pd.Timestamp(split.data_start)
        if mode == "rolling":
            lo = max(lo, test_start - pd.DateOffset(years=train_years))
        train = calendar.sessions(lo, test_start - pd.Timedelta(days=1))
        if len(train) < 250:
            raise ValueError(
                f"fold for {year}: fewer than 250 training sessions after {split.data_start}"
            )
        phase_a, phase_b = split.phase_of(test_start), split.phase_of(test_end)
        if phase_a != phase_b:
            raise ValueError(f"test block {year} straddles phases {phase_a}/{phase_b}")
        folds.append(Fold(i, train[0], train[-1], test_start, test_end, phase_a))
    return folds


@dataclass
class FoldData:
    """What a fold needs, loaded once: the screen (training data only) and prices to test_end."""

    fold: Fold
    screen: ScreenResult
    logp: pd.DataFrame
    ret: pd.DataFrame
    event: pd.DataFrame  # bool (session x ticker): non-ordinary distribution on that ex-date
    dollar_volume: pd.DataFrame | None = None  # close x volume, split-invariant (for cost models)
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    close: pd.DataFrame | None = None


def prepare_fold(
    pipe, fold: Fold, screen_cfg: ScreenConfig, max_dividend_yield: float = 0.10
) -> FoldData:
    """Screen on the training window, then load prices for the whole fold (as of ``test_end``)."""
    screen = discover_pairs(pipe, fold.train_start, fold.train_end, screen_cfg)
    tickers = sorted(set(screen.pairs["y"]) | set(screen.pairs["x"]))
    panel = pipe.panel(tickers, fold.train_start, fold.test_end)
    event = (panel.dividends / panel.close.shift(1)) > max_dividend_yield
    return FoldData(
        fold,
        screen,
        np.log(panel.tr_close()),
        panel.ret,
        event.fillna(False),
        panel.dollar_volume,
        panel.high,
        panel.low,
        panel.close,
    )


def blackout_mask(event: np.ndarray, lead: int, after: int) -> np.ndarray:
    """True from ``lead`` bars before to ``after`` bars after each event bar."""
    n = len(event)
    mask = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(event):
        mask[max(0, i - lead) : min(n, i + after + 1)] = True
    return mask


@dataclass
class PairRun:
    y: str
    x: str
    frame: pd.DataFrame  # pnl, cost, trade, gross ... over the TEST block only
    trades: pd.DataFrame


def run_pairs(
    fd: FoldData,
    strat: StrategyConfig,
    pairs: Iterable[tuple[str, str]],
    cost_fn: Callable[[pd.DataFrame, str, str], pd.Series | pd.DataFrame] | None = None,
) -> list[PairRun]:
    """Trade each (y, x) with the fold's frozen hedge ratio over the test block."""
    fold, out = fd.fold, []
    after = strat.event_blackout if strat.event_blackout is not None else strat.z_window
    for y, x in pairs:
        ev = (fd.event[y] | fd.event[x]).to_numpy()
        run = run_pair(
            fd.logp[y],
            fd.logp[x],
            fd.ret[y],
            fd.ret[x],
            strat.hedge,
            strat.z_window,
            strat.params,
            fold.train_end,
            target_vol=strat.target_vol,
            freeze_beta=strat.freeze_beta,
            blackout=blackout_mask(ev, strat.event_lead, after),
            flatten_at_end=True,
            on_missing_return="zero",
        )
        frame = run["returns"].loc[fold.test_start : fold.test_end].copy()
        if cost_fn is None:
            frame["cost"] = 0.0
        else:
            c = cost_fn(frame, y, x)
            if isinstance(c, pd.DataFrame):
                for col in c.columns:
                    frame[col] = c[col].reindex(frame.index).to_numpy()
            else:
                frame["cost"] = np.asarray(c, dtype=float)
        frame["net"] = frame["pnl"] - frame["cost"]
        tr = run["trades"]
        tr = tr[tr["entry"] >= fold.test_start].assign(y=y, x=x)
        out.append(PairRun(y, x, frame, tr))
    return out


def aggregate(runs: list[PairRun], index: pd.DatetimeIndex, slots: int) -> pd.DataFrame:
    """Fixed-slot equal-weight portfolio: sum of pair P&L / ``slots`` (return on total capital)."""
    cols = ["pnl", "cost", "net", "trade", "gross"]
    cols += sorted({c for r in runs for c in r.frame.columns if c.startswith("cost_")})
    total = pd.DataFrame(0.0, index=index, columns=cols)
    opened = pd.Series(0.0, index=index)
    for r in runs:
        total += r.frame[cols].reindex(index).fillna(0.0)
        opened += (r.frame["gross"].reindex(index).fillna(0.0) > 0).astype(float)
    total = total / slots
    total["n_open"] = opened
    total["missing_days"] = sum(
        (r.frame["missing"].reindex(index).fillna(False).astype(int) for r in runs), 0
    )
    return total


@dataclass
class WalkForwardResult:
    daily: (
        pd.DataFrame
    )  # stitched out-of-sample series (net/gross return on capital, turnover, ...)
    trades: pd.DataFrame
    folds: pd.DataFrame  # per-fold bookkeeping
    config: dict = field(default_factory=dict)


def walk_forward(
    fold_data: list[FoldData],
    strat: StrategyConfig,
    selector: Literal["screen", "random"] = "screen",
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
    cost_fn: Callable[[pd.DataFrame, str, str], pd.Series | pd.DataFrame] | None = None,
    allowed_phases: tuple[str, ...] = ("research",),
    cost_factory: Callable[[FoldData], Callable] | None = None,
) -> WalkForwardResult:
    """Trade every fold's selected pairs (placebo: random same-count non-significant pairs).

    Costs come from ``cost_fn`` (one function for all folds) or ``cost_factory(fold_data)`` (a cost
    function built per fold, as a cost model needs each fold's own market data); not both.
    """
    if cost_fn is not None and cost_factory is not None:
        raise ValueError("pass cost_fn or cost_factory, not both")
    bad = {fd.fold.phase for fd in fold_data} - set(allowed_phases)
    if bad:
        raise ValueError(
            f"folds in phases {sorted(bad)} are not allowed here (allowed: {allowed_phases})"
        )
    parts, trade_parts, meta = [], [], []
    for fd in fold_data:
        pairs = fd.screen.pairs
        chosen = pairs[pairs["selected"]]
        if selector == "random":
            if rng is None:
                raise ValueError("selector='random' needs an rng")
            pool = pairs[(pairs["p_calibrated"] > alpha) & (pairs["beta"] > 0)]
            k = min(len(chosen), len(pool))
            chosen = pool.iloc[rng.choice(len(pool), k, replace=False)] if k else pool.iloc[0:0]
        index = fd.logp.loc[fd.fold.test_start : fd.fold.test_end].index
        fold_cost = cost_factory(fd) if cost_factory is not None else cost_fn
        runs = run_pairs(fd, strat, list(zip(chosen["y"], chosen["x"], strict=True)), fold_cost)
        daily = aggregate(runs, index, strat.slots)
        daily["fold"] = fd.fold.index
        parts.append(daily)
        trade_parts += [r.trades for r in runs if len(r.trades)]
        meta.append(
            {
                "fold": fd.fold.index,
                "phase": fd.fold.phase,
                "train_start": fd.fold.train_start,
                "train_end": fd.fold.train_end,
                "test_start": fd.fold.test_start,
                "test_end": fd.fold.test_end,
                "n_eligible": fd.screen.n_eligible,
                "n_candidates": fd.screen.n_candidates,
                "n_selected": len(chosen),
                "pairs": list(zip(chosen["y"], chosen["x"], strict=True)),
                "missing_days": int(daily["missing_days"].sum()),
            }
        )
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    return WalkForwardResult(pd.concat(parts), trades, pd.DataFrame(meta), strat.model_dump())
