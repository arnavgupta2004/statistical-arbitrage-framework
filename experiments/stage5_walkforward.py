"""Stage 5 experiment: walk-forward on the RESEARCH phase only (test years 2015-2018), gross.

Fixed before running.  Validation (2019-2021) and the holdout (2022+) are not touched: this
stage builds the engine and answers Experiment A with a placebo; configuration choice with costs
is Stage 6, scored once on validation.

Protocol
    Folds        4 calendar-year test blocks, 2015 / 2016 / 2017 / 2018, each screened on the four
                 years before it (rolling 4-year window; the first starts 2011-01-03), positions
                 flat at each block's end.  Point-in-time S&P 500 members, identity-verified.
    Screen       default ScreenConfig (k = 5, alpha 0.05, half-life 5-60 days, beta > 0, 20 slots,
                 10 shifted-null panels), seed fixed.  Screening does not depend on the strategy
                 configuration, so each fold is screened once and reused.
    Grid (9)     hedge in {static, expanding, kalman(delta 1e-5)}  x  entry in {1.5, 2.0, 2.5};
                 exit 0.5, stop 4.0, max hold 60, z-window 60, unit sizing, daily re-hedging,
                 no costs.
                 (The grid was pruned to these hedge methods using Stage 4's simulator and 2016
                 results, which are research-phase data; all 9 are registered as trials.)
    Placebo      the same strategy on RANDOM pairs: for each fold, the same number of pairs as the
                 screen selected, drawn from that fold's non-significant candidates with beta > 0
                 (same sector, same correlation filter, no cointegration evidence).  300 draws; the
                 draws are identical across configurations, so comparisons are paired.

Hypotheses
    H5a  The screened portfolio's gross Sharpe exceeds the placebo distribution (percentile >= 95 %)
         for the best configuration.  Prior, from Stages 3-4: it does not.
    H5b  The strategy itself (placebo) earns a positive gross Sharpe on the research blocks.
    H5c  The ordering of the hedge methods echoes the simulator (static ~ expanding >= Kalman).

Everything is gross: the daily series are saved so Stage 6 can charge costs on the same trades.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage5_walkforward
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from statarb.backtest import metrics
from statarb.backtest.walkforward import (
    StrategyConfig,
    make_folds,
    prepare_fold,
    run_pairs,
    walk_forward,
)
from statarb.config import load_config
from statarb.data.pipeline import DataPipeline
from statarb.research.registry import Registry
from statarb.selection.screening import ScreenConfig
from statarb.statistics.bootstrap import bootstrap_ci

SEED = 20260922
FIRST_YEAR, LAST_YEAR, TRAIN_YEARS = 2015, 2018, 4
N_DRAWS = 300
GRID = [
    StrategyConfig(hedge_method=h, kalman_delta=1e-5, entry=e)
    for h in ("static", "expanding", "kalman")
    for e in (1.5, 2.0, 2.5)
]
OUT = Path("experiments/results")


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def pool_pairs(fd, alpha: float = 0.05) -> list[tuple[str, str]]:
    p = fd.screen.pairs
    pool = p[(p["p_calibrated"] > alpha) & (p["beta"] > 0)]
    return list(zip(pool["y"], pool["x"], strict=True))


def placebo_series(fds, strat: StrategyConfig, draws: dict) -> np.ndarray:
    """(N_DRAWS, total_days) matrix of placebo daily P&L, sharing the draws across configs."""
    blocks = []
    for fd in fds:
        runs = run_pairs(fd, strat, pool_pairs(fd))
        mat = np.vstack([r.frame["pnl"].to_numpy() for r in runs])
        idx = draws[fd.fold.index]  # (N_DRAWS, k) row indices into the pool
        blocks.append(mat[idx].sum(axis=1) / strat.slots)  # (N_DRAWS, T_fold)
        del runs, mat
    return np.hstack(blocks)


def main() -> None:
    warnings.filterwarnings("ignore")
    pipe = DataPipeline(load_config("configs/data_default.yaml"))
    split = pipe.cfg.split
    folds = make_folds(split, pipe.calendar, FIRST_YEAR, LAST_YEAR, TRAIN_YEARS, "rolling")
    assert all(f.phase == "research" for f in folds), "Stage 5 evaluates the research phase only"
    screen_cfg = ScreenConfig(seed=SEED, n_null_panels=10)
    fds = []
    for f in folds:
        print(
            f"fold {f.index}: train {f.train_start.date()}..{f.train_end.date()} "
            f"test {f.test_start.date()}..{f.test_end.date()}",
            flush=True,
        )
        fds.append(prepare_fold(pipe, f, screen_cfg))
        s = fds[-1].screen
        print(
            f"   eligible {s.n_eligible}, candidates {s.n_candidates}, "
            f"selected {int(s.pairs['selected'].sum())}, "
            f"calibrated discoveries@5% {s.discoveries(0.05)}, est FDR {s.estimated_fdr(0.05):.2f}",
            flush=True,
        )

    draws = {}
    for fd in fds:
        k = int(fd.screen.pairs["selected"].sum())
        n_pool = len(pool_pairs(fd))
        draws[fd.fold.index] = np.vstack(
            [
                np.random.default_rng([SEED, fd.fold.index, d]).choice(
                    n_pool, size=k, replace=False
                )
                for d in range(N_DRAWS)
            ]
        )

    registry = Registry()
    fingerprint = pipe.fingerprint()
    results, daily_parts = {}, []
    for strat in GRID:
        wf = walk_forward(fds, strat)
        d = wf.daily
        r = d["net"].to_numpy()
        sh, ci, _ = bootstrap_ci(r, metrics.sharpe, n_boot=2000, mean_block=10, seed=1)
        plc = placebo_series(fds, strat, draws)
        p_sh = np.array([metrics.sharpe(x) for x in plc])
        p_mean = plc.mean(axis=1)
        summ = metrics.performance_summary(r, d["trade"].to_numpy())
        by_fold = {int(f): float(metrics.sharpe(g["net"].to_numpy())) for f, g in d.groupby("fold")}
        results[strat.label] = {
            "config": strat.model_dump(),
            "summary": summ,
            "sharpe_ci95": list(ci),
            "sharpe_by_fold": by_fold,
            "n_trades": int(len(wf.trades)),
            "stop_share": float((wf.trades["exit_reason"] == "stop").mean())
            if len(wf.trades)
            else None,
            "placebo": {
                "n_draws": N_DRAWS,
                "sharpe_median": float(np.median(p_sh)),
                "sharpe_5_95": [float(np.quantile(p_sh, 0.05)), float(np.quantile(p_sh, 0.95))],
                "screened_percentile": float((p_sh < sh).mean()),
                "mean_daily_bps_median": float(np.median(p_mean) * 1e4),
                "screened_minus_placebo_mean_bps": float((d["net"].mean() - p_mean.mean()) * 1e4),
                "share_of_draws_with_positive_sharpe": float((p_sh > 0).mean()),
            },
        }
        x = results[strat.label]
        print(
            f"{strat.label:16s} Sharpe {sh:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] | placebo median "
            f"{x['placebo']['sharpe_median']:+.2f} (5-95%: {x['placebo']['sharpe_5_95'][0]:+.2f},"
            f"{x['placebo']['sharpe_5_95'][1]:+.2f}) | "
            f"screened at pctile {x['placebo']['screened_percentile']:.0%} "
            f"| turnover/yr {summ['turnover_per_year']:.1f} trades {x['n_trades']}",
            flush=True,
        )
        daily_parts.append(d.assign(config=strat.label))
        registry.register(
            stage=5,
            kind="strategy",
            name=f"walkforward_{strat.label}",
            phases=["research"],
            windows={
                "test_years": [FIRST_YEAR, LAST_YEAR],
                "train_years": TRAIN_YEARS,
                "mode": "rolling",
            },
            universe="S&P 500 point-in-time, identity-verified",
            strategy="screened pairs, z-score reversion",
            parameters={"strategy": strat.model_dump(), "screen": screen_cfg.model_dump()},
            features=["log dividend-adjusted price", "closed-form z-score"],
            results={"gross_sharpe": sh, "placebo_percentile": x["placebo"]["screened_percentile"]},
            seed=SEED,
            data_fingerprint=fingerprint,
            git_commit=git_commit(),
            notes="gross, research phase only; validation and holdout untouched",
        )

    fold_table = pd.DataFrame(
        [
            {
                "fold": fd.fold.index,
                "train": [str(fd.fold.train_start.date()), str(fd.fold.train_end.date())],
                "test": [str(fd.fold.test_start.date()), str(fd.fold.test_end.date())],
                "eligible": fd.screen.n_eligible,
                "candidates": fd.screen.n_candidates,
                "selected": int(fd.screen.pairs["selected"].sum()),
                "discoveries_5pct": fd.screen.discoveries(0.05),
                "estimated_fdr_5pct": fd.screen.estimated_fdr(0.05),
            }
            for fd in fds
        ]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(daily_parts).to_csv(OUT / "stage5_daily_returns.csv", index_label="date")
    out = {
        "experiment": "stage5_walkforward",
        "seed": SEED,
        "git_commit": git_commit(),
        "data_fingerprint": fingerprint,
        "split": split.model_dump(mode="json"),
        "folds": fold_table.to_dict(orient="records"),
        "results": results,
        "phases_evaluated": ["research"],
        "note": "gross of costs; validation and holdout untouched",
    }
    (OUT / "stage5_walkforward.json").write_text(json.dumps(out, indent=2, default=str))
    print("wrote", OUT / "stage5_walkforward.json")
    pipe.close()


if __name__ == "__main__":
    sys.exit(main())
