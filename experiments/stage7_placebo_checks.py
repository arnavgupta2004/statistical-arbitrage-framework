"""Stage 7 checks of the label-permutation placebo (the null used for the PCA family).

A.  Calibration under a true null (synthetic).  Returns = factors + the change of a *random-walk*
    idiosyncratic level: there is no residual mean reversion to find.  For each of 40 seeds the
    real book's gross P&L is ranked against 60 placebo draws.  A calibrated null gives ranks
    spread over (0, 1), rejects at about the nominal rate, and the placebo's dispersion matches the
    dispersion of the real book across seeds.
B.  Real-data comparability.  For ``pca_reversal_k5`` on the four research folds: does the placebo
    book trade and pay the same as the real book (turnover, gross exposure, each cost component)?
    Any asymmetry would bias the net comparison.

Pre-specified reading: A passes if the mean rank is within 0.5 +- 0.1, the share of ranks beyond the
5 % / 95 % tails is within 0.05 +- 0.06 each side, and the dispersion ratio is within 0.75-1.33.
B is descriptive: the placebo/real ratios are reported, not thresholded.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage7_placebo_checks
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from statarb.backtest.costs import CostConfig
from statarb.backtest.pca_walkforward import (
    FoldSignals,
    PCAStrategyConfig,
    fold_signals,
    placebo_permutation,
    prepare_pca_fold,
    run_fold,
)
from statarb.backtest.walkforward import make_folds
from statarb.config import load_config
from statarb.data.pipeline import DataPipeline
from statarb.research.registry import Registry
from statarb.selection.screening import ScreenConfig
from statarb.signals.residuals import ResidualConfig, residual_signals

OUT = Path("experiments/results")
N_SEEDS, N_DRAWS = 40, 60
CFG = PCAStrategyConfig(
    residual=ResidualConfig(
        n_factors=2, fit_window=250, refit_every=21, beta_window=40, signal="reversal"
    )
)


def factor_world(phi: float, seed: int, t: int = 700, n: int = 40) -> FoldSignals:
    """Factors plus the change in an AR(1) idiosyncratic level (phi = 1: a random walk)."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, [0.010, 0.006], (t, 2))
    b = rng.normal(0, 1, (n, 2))
    b[:, 0] = np.abs(b[:, 0]) + 0.5
    x = np.zeros((t, n))
    for j in range(1, t):
        x[j] = phi * x[j - 1] + rng.normal(0, 0.01, n)
    idx = pd.bdate_range("2019-01-01", periods=t)
    r = pd.DataFrame(f @ b.T + np.diff(x, axis=0, prepend=0.0), index=idx)
    sig = residual_signals(r, 300, CFG.residual)
    d = len(sig.dates)
    market = {
        k: np.full((d, n), v) for k, v in (("adv", 1e10), ("sigma", 0.01), ("half_spread", 1e-4))
    }
    return FoldSignals(
        None, sig, market, r.to_numpy()[300:], np.ones((d, n), bool), np.zeros((d, n), bool)
    )


def calibration(phi: float) -> dict:
    ranks, reals, sds = [], [], []
    for seed in range(N_SEEDS):
        fs = factor_world(phi, 100 + seed)
        real = run_fold(fs, CFG)["pnl"].sum()
        rng = np.random.default_rng(seed)
        plc = np.array(
            [
                run_fold(fs, CFG, perm=placebo_permutation(fs, rng))["pnl"].sum()
                for _ in range(N_DRAWS)
            ]
        )
        ranks.append(float((plc < real).mean()))
        reals.append(float(real))
        sds.append(float(plc.std()))
    ranks = np.array(ranks)
    return {
        "phi": phi,
        "n_seeds": N_SEEDS,
        "n_draws": N_DRAWS,
        "mean_rank": float(ranks.mean()),
        "share_rank_below_5pct": float((ranks < 0.05).mean()),
        "share_rank_above_95pct": float((ranks > 0.95).mean()),
        "rank_histogram_quintiles": np.histogram(ranks, bins=5, range=(0, 1))[0].tolist(),
        "placebo_sd_over_real_sd": float(np.mean(sds) / np.std(reals)),
        "mean_real_pnl": float(np.mean(reals)),
    }


def real_data_symmetry() -> dict:
    pipe = DataPipeline(load_config("configs/data_default.yaml"))
    folds = make_folds(pipe.cfg.split, pipe.calendar, 2015, 2018, 4, "rolling")
    cost = CostConfig(capital=1e8, slots=20)
    cfg = PCAStrategyConfig(residual=ResidualConfig(n_factors=5, signal="reversal"))
    rows = []
    for f in folds:
        fs = fold_signals(prepare_pca_fold(pipe, f, ScreenConfig()), cfg, cost)

        def summ(x):
            return {
                "turnover": x.trade.sum(),
                "gross_exposure": x.gross_exposure.mean(),
                "cost_spread": x.cost_spread.sum(),
                "cost_commission": x.cost_commission.sum(),
                "cost_impact": x.cost_impact.sum(),
                "cost_borrow": x.cost_borrow.sum(),
                "std_daily_pnl": x.pnl.std(),
                "std_daily_net": x.net.std(),
            }

        real = summ(run_fold(fs, cfg, cost))
        plc = pd.DataFrame(
            [
                summ(run_fold(fs, cfg, cost, placebo_permutation(fs, np.random.default_rng(d))))
                for d in range(30)
            ]
        ).mean()
        rows.append(pd.DataFrame({"real": real, "placebo": plc}))
    pipe.close()
    m = pd.concat(rows).groupby(level=0).mean()
    m["placebo_over_real"] = m["placebo"] / m["real"]
    return m.to_dict(orient="index")


def main() -> None:
    warnings.filterwarnings("ignore")
    out = {
        "experiment": "stage7_placebo_checks",
        "A_null_calibration": calibration(phi=1.0),
        "B_real_data_symmetry_pca_reversal_k5": real_data_symmetry(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage7_placebo_checks.json").write_text(json.dumps(out, indent=2))
    a = out["A_null_calibration"]
    print(
        f"A: mean rank {a['mean_rank']:.3f}, tails <5% {a['share_rank_below_5pct']:.2f} / "
        f">95% {a['share_rank_above_95pct']:.2f}, placebo sd / real sd {a['placebo_sd_over_real_sd']:.2f}"
    )
    for k, v in out["B_real_data_symmetry_pca_reversal_k5"].items():
        print(f"B: {k:16s} placebo/real {v['placebo_over_real']:.3f}")
    Registry().register(
        stage=7,
        kind="simulation",
        name="pca_placebo_checks",
        phases=["research"],
        strategy="none (checks of the placebo null)",
        parameters={"n_seeds": N_SEEDS, "n_draws": N_DRAWS},
        notes="A is synthetic; B re-runs pca_reversal_k5 on the research folds (descriptive)",
    )


if __name__ == "__main__":
    sys.exit(main())
