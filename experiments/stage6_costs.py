"""Stage 6 experiment: how much of the Stage 5 gross result survives realistic costs?

Fixed before running.  It re-scores the SAME nine configurations on the SAME four research folds
(2015-2018, screens reproduced from the same seed) with the cost model of ``backtest/costs.py``.
Validation (2019-2021) and the holdout (2022+) are not touched.

Central cost case (the one that matters; everything else is sensitivity)
    capital $100 M over 20 slots ($5 M per pair slot); half-spread by trailing dollar ADV (1.0 bp
    above $500 M, 1.5 bp $100-500 M, 2.5 bp $25-100 M, else 5 bp); commission 0.5 bp + regulatory
    0.25 bp on all traded notional; short borrow 50 bps a year; impact Y sigma sqrt(Q / ADV) with
    Y = 0.5, ADV a trailing 20-day median, sigma a trailing 20-day volatility (both lagged a day).

Sensitivity (screened portfolios only)
    cost waterfall (spread -> commission -> borrow -> impact); impact Y in {0, 0.25, 0.5, 1, 2} x
    capital in {$10 M, $100 M, $1 B}; spread model in {tiered, Corwin-Schultz, max(tiered, CS)} and
    spread multiplier in {0, 0.5, 1, 2}; borrow in {0, 50, 150} bps.  Breakeven cost multiple: the
    factor on ALL costs at which the net mean return reaches zero.

Placebo: the Stage 5 placebo (300 draws of random non-significant, beta > 0 pairs, same count,
identical draws across configurations) is re-scored NET with the central costs.

FINALIST RULE, written before any net result was seen
    A configuration advances to the validation phase only if, over the research folds and with
    central costs, (1) its net Sharpe is positive AND (2) its net Sharpe is at or above the median of
    its own net placebo (percentile >= 50 %).  The finalist is the qualifying configuration with the
    highest net Sharpe.  If none qualifies, NOTHING advances and validation is not run: an honest
    outcome, not a failure to be tuned away.

Hypotheses
    H6a  Costs remove the gross edge: central-cost net Sharpe < gross Sharpe for all nine, and the
         breakeven cost multiple is below 1 for most.
    H6b  Turnover ranks the configurations' cost drag (entry 1.5, the most active, suffers most).
    H6c  At a small capital ($10 M) impact is minor and the spread/commission pieces dominate; at
         $1 B impact dominates.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage6_costs
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.stage5_walkforward import (
    FIRST_YEAR,
    GRID,
    LAST_YEAR,
    N_DRAWS,
    SEED,
    TRAIN_YEARS,
    pool_pairs,
)
from statarb.backtest import metrics
from statarb.backtest.costs import CostConfig, PairCostModel, breakeven_multiple, build_market_data
from statarb.backtest.walkforward import make_folds, prepare_fold, run_pairs, walk_forward
from statarb.config import load_config
from statarb.data.pipeline import DataPipeline
from statarb.research.registry import Registry
from statarb.selection.screening import ScreenConfig
from statarb.statistics.bootstrap import bootstrap_ci

OUT = Path("experiments/results")
IMPACT_Y = (0.0, 0.25, 0.5, 1.0, 2.0)
CAPITALS = (1e7, 1e8, 1e9)
COMPONENTS = ("cost_spread", "cost_commission", "cost_borrow", "cost_impact")


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def factory(cfg: CostConfig):
    """One cost model per fold (the market data of that fold's tickers and dates)."""

    def make(fd):
        m = build_market_data(fd.dollar_volume, fd.ret, fd.high, fd.low, fd.close, cfg)
        return PairCostModel(m, cfg)

    return make


def net_stats(d: pd.DataFrame) -> dict:
    s = metrics.performance_summary(d["net"].to_numpy(), d["trade"].to_numpy())
    return {
        "net_sharpe": s["sharpe"],
        "net_ann_return_pct": s["ann_return_pct"],
        "max_drawdown_pct": s["max_drawdown_pct"],
        "turnover_per_year": s["turnover_per_year"],
    }


def participation_stats(fds, strat, models) -> dict:
    """Trade-day participation (Q / ADV, larger leg) of the screened pairs under the central cost."""
    parts = []
    for fd in fds:
        sel = fd.screen.selected
        runs = run_pairs(
            fd, strat, list(zip(sel["y"], sel["x"], strict=True)), models[fd.fold.index]
        )
        for r in runs:
            traded = r.frame["trade"] > 0
            parts.append(r.frame.loc[traded, "participation"].to_numpy())
    p = np.concatenate(parts) if parts else np.array([0.0])
    return {
        "n_trade_days": int(len(p)),
        "median": float(np.median(p)),
        "p90": float(np.quantile(p, 0.9)),
        "share_above_5pct": float((p > 0.05).mean()),
        "share_above_10pct": float((p > 0.10).mean()),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    pipe = DataPipeline(load_config("configs/data_default.yaml"))
    folds = make_folds(pipe.cfg.split, pipe.calendar, FIRST_YEAR, LAST_YEAR, TRAIN_YEARS, "rolling")
    assert all(f.phase == "research" for f in folds)
    screen_cfg = ScreenConfig(seed=SEED, n_null_panels=10)
    fds = [prepare_fold(pipe, f, screen_cfg) for f in folds]
    print("folds prepared", flush=True)

    # ---- reproduction of Stage 5's gross series (a pipeline-stability check) ----------------------
    saved = pd.read_csv(OUT / "stage5_daily_returns.csv", parse_dates=["date"])
    worst = 0.0
    for strat in GRID:
        d = walk_forward(fds, strat).daily
        ref = saved[saved.config == strat.label].set_index("date")
        worst = max(
            worst,
            float((d["pnl"] - ref["pnl"]).abs().max()),
            float((d["trade"] - ref["trade"]).abs().max()),
        )
    print(f"gross series reproduced against Stage 5: max |diff| = {worst:.2e}", flush=True)
    assert worst < 1e-10, (
        "fold screens no longer reproduce Stage 5; results would not be comparable"
    )

    registry = Registry()
    central = CostConfig()
    fingerprint = pipe.fingerprint()
    draws = {}
    for fd in fds:  # identical to Stage 5
        k = int(fd.screen.pairs["selected"].sum())
        n_pool = len(pool_pairs(fd))
        draws[fd.fold.index] = np.vstack(
            [
                np.random.default_rng([SEED, fd.fold.index, i]).choice(
                    n_pool, size=k, replace=False
                )
                for i in range(N_DRAWS)
            ]
        )
    models = {fd.fold.index: factory(central)(fd) for fd in fds}

    results, daily_parts = {}, []
    for strat in GRID:
        lab = strat.label
        d = walk_forward(fds, strat, cost_factory=factory(central)).daily
        net = d["net"].to_numpy()
        sh_net, ci, _ = bootstrap_ci(net, metrics.sharpe, n_boot=2000, mean_block=10, seed=1)
        sh_gross = metrics.sharpe(d["pnl"].to_numpy())
        blocks = []
        for fd in fds:  # net placebo with the same draws as Stage 5
            runs = run_pairs(fd, strat, pool_pairs(fd), models[fd.fold.index])
            mat = np.vstack([r.frame["net"].to_numpy() for r in runs])
            blocks.append(mat[draws[fd.fold.index]].sum(axis=1) / strat.slots)
            del runs, mat
        p_sh = np.array([metrics.sharpe(x) for x in np.hstack(blocks)])
        mean_gross, mean_cost = float(d["pnl"].mean()), float(d["cost"].mean())
        total_cost = float(d["cost"].sum())
        traded_units = float(d["trade"].sum())  # capital units, already divided by slots
        results[lab] = {
            "config": strat.model_dump(),
            "gross": {
                "sharpe": sh_gross,
                "mean_daily_bps": mean_gross * 1e4,
                "ann_return_pct": mean_gross * 252 * 100,
            },
            "net": {
                **net_stats(d),
                "sharpe_ci95": list(ci),
                "mean_daily_bps": float(d["net"].mean() * 1e4),
            },
            "cost": {
                "mean_daily_bps": mean_cost * 1e4,
                "ann_pct": mean_cost * 252 * 100,
                "share_of_gross_pct": 100 * mean_cost / mean_gross if mean_gross > 0 else None,
                "components_share": {c: float(d[c].sum() / total_cost) for c in COMPONENTS},
                "bps_per_unit_traded": 1e4 * total_cost / traded_units,
                "breakeven_multiple": breakeven_multiple(mean_gross, mean_cost),
            },
            "participation": participation_stats(fds, strat, models),
            "placebo_net": {
                "sharpe_median": float(np.median(p_sh)),
                "sharpe_5_95": [float(np.quantile(p_sh, 0.05)), float(np.quantile(p_sh, 0.95))],
                "screened_percentile": float((p_sh < sh_net).mean()),
            },
        }
        x = results[lab]
        print(
            f"{lab:16s} gross {sh_gross:+.2f} -> net {sh_net:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] | cost "
            f"{x['cost']['ann_pct']:.2f}%/yr vs gross {x['gross']['ann_return_pct']:.2f}%/yr | "
            f"breakeven x{x['cost']['breakeven_multiple']:.2f} | net placebo median "
            f"{x['placebo_net']['sharpe_median']:+.2f}, pctile {x['placebo_net']['screened_percentile']:.0%}",
            flush=True,
        )
        daily_parts.append(d.assign(config=lab))
        registry.register(
            stage=6,
            kind="diagnostic",
            name=f"cost_rescore_{lab}",
            phases=["research"],
            windows={"test_years": [FIRST_YEAR, LAST_YEAR], "train_years": TRAIN_YEARS},
            strategy="screened pairs, z-score reversion",
            parameters={"strategy": strat.model_dump()},
            costs=central.model_dump(),
            results={"net_sharpe": sh_net, "gross_sharpe": sh_gross},
            seed=SEED,
            data_fingerprint=fingerprint,
            git_commit=git_commit(),
            notes="re-scoring of a registered Stage 5 strategy trial with costs; not a new configuration",
        )

    # ---- sensitivity, screened portfolios only --------------------------------------------------------
    sens: dict = {"impact_y_x_capital": {}, "spread": {}, "borrow": {}, "waterfall": {}}
    for strat in GRID:
        lab = strat.label

        def run(cfg, strat=strat):
            return walk_forward(fds, strat, cost_factory=factory(cfg)).daily

        for cap in CAPITALS:
            for y in IMPACT_Y:
                sens["impact_y_x_capital"][f"{lab}|{cap:.0e}|{y}"] = net_stats(
                    run(CostConfig(capital=cap, impact_y=y))
                )
        for model in ("tiered", "corwin_schultz", "max"):
            for mult in (0.0, 0.5, 1.0, 2.0):
                cfg = CostConfig(spread_model=model, spread_mult=mult)
                sens["spread"][f"{lab}|{model}|{mult}"] = net_stats(run(cfg))
        for b in (0.0, 50.0, 150.0):
            sens["borrow"][f"{lab}|{b}"] = net_stats(run(CostConfig(borrow_bps_per_year=b)))
        dc = run(central)
        cum = dc["pnl"].to_numpy().copy()
        steps = {"gross": metrics.sharpe(cum)}
        for c in COMPONENTS:
            cum = cum - dc[c].to_numpy()
            steps[c.replace("cost_", "+")] = metrics.sharpe(cum)
        sens["waterfall"][lab] = steps
        print("sensitivity done", lab, flush=True)

    # ---- finalist rule (pre-specified) ---------------------------------------------------------------
    qualifying = {
        k: v
        for k, v in results.items()
        if v["net"]["net_sharpe"] > 0 and v["placebo_net"]["screened_percentile"] >= 0.5
    }
    finalist = (
        max(qualifying, key=lambda k: qualifying[k]["net"]["net_sharpe"]) if qualifying else None
    )
    print(
        "qualifying configurations:",
        list(qualifying) or "none",
        "| finalist:",
        finalist,
        flush=True,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(daily_parts).to_csv(OUT / "stage6_daily_net_returns.csv", index_label="date")
    out = {
        "experiment": "stage6_costs",
        "seed": SEED,
        "git_commit": git_commit(),
        "data_fingerprint": fingerprint,
        "central_cost": central.model_dump(),
        "gross_reproduction_max_abs_diff": worst,
        "results": results,
        "sensitivity": sens,
        "finalist_rule": (
            "net Sharpe > 0 AND net-placebo percentile >= 50 %; highest net Sharpe among qualifiers; "
            "none -> no configuration advances and validation is not run"
        ),
        "qualifying": list(qualifying),
        "finalist": finalist,
        "phases_evaluated": ["research"],
    }
    (OUT / "stage6_costs.json").write_text(json.dumps(out, indent=2, default=str))
    registry.register(
        stage=6,
        kind="diagnostic",
        name="cost_sensitivity_grid",
        phases=["research"],
        parameters={
            "impact_y": IMPACT_Y,
            "capitals": CAPITALS,
            "borrow": [0, 50, 150],
            "spread": "3 models x 4 multipliers",
        },
        results={"finalist": finalist, "n_qualifying": len(qualifying)},
        seed=SEED,
        data_fingerprint=fingerprint,
        git_commit=git_commit(),
        notes="sensitivity of the nine Stage 5 configurations; no configuration selected from it",
    )
    print("wrote", OUT / "stage6_costs.json")
    pipe.close()


if __name__ == "__main__":
    sys.exit(main())
