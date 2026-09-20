"""Stage 4 experiment: how do hedge-ratio methods behave when the truth is KNOWN?

Real data cannot say which hedge estimator is right, because the true hedge ratio is unobserved.  A
simulator with a known beta_t can.  Everything below was fixed before running.

Data-generating process (n = 1750 days; the first 1250 are "training", the last 500 the test)::

    lx_t = 4 + random walk (drift 3e-4, sd 1.5e-2)
    u_t  = OU spread noise, kappa = 0.08 (half-life 8.7 days), stationary sd 0.03
    ly_t = 0.3 + beta_t * lx_t + u_t

    R1  constant      beta_t = 0.8
    R2  drifting      beta_t = 0.8 + random walk, step sd 0.002 (test-period drift sd ~0.045)
    R3  break         beta_t = 0.8, jumping to 1.05 at the middle of the test period

Methods (all causal; static is fitted on the first 1250 days only)::

    static, expanding, rolling 60 / 120 / 250, kalman delta in {1e-6, 1e-5, 1e-4}

Trading rule for all: z-window 60, entry 2.0, exit 0.5, stop 4.0, max hold 60, unit sizing, daily
re-hedging.  Metrics per method and regime: RMSE of the estimated beta over the test period,
gross annualised Sharpe, mean daily P&L, turnover, trades and stop-out share.  200 simulations per
regime; intervals are +/- 1.96 standard errors across simulations.

Hypotheses
    HS1  R1: static and expanding estimate beta better than short rolling windows and Kalman
         (less estimation noise when nothing moves).
    HS2  R2: rolling / Kalman estimate beta better than static (they can follow the drift).
    HS3  R3: after a break rolling / Kalman recover; static's error stays at the size of the jump.
    HS4  No method is best in every regime (a method's worst-regime rank is reported).
    HS5  The lower beta error of an adaptive method does not necessarily translate into a higher
         Sharpe once turnover is counted (reported, not assumed).

After seeing the first run a POST-HOC regime was added, clearly labelled and written to its own file
(``--posthoc``): R3b, a mild break (beta 0.8 -> 0.85, a spread level shift of ~0.2, about 7
stationary sd, against ~33 sd for the pre-specified R3, which is larger than a realistic break).
It was added because R3's step wrecks any window that contains it; it is explanatory, not a
replacement.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage4_hedge_ratio_synthetic [--posthoc]
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from statarb.backtest.pair_pnl import run_pair
from statarb.models.hedge_ratio import HedgeSpec, hedge_path
from statarb.models.ou import simulate_ou
from statarb.signals.pairs import PairParams
from statarb.statistics.bootstrap import sharpe

SEED = 20260921
N, N_TRAIN, SIMS = 1750, 1250, 200
KAPPA, SD_U = 0.08, 0.03
PARAMS, Z_WINDOW = PairParams(2.0, 0.5, 4.0, 60), 60
SPECS = (
    [HedgeSpec("static"), HedgeSpec("expanding")]
    + [HedgeSpec("rolling", window=w) for w in (60, 120, 250)]
    + [HedgeSpec("kalman", delta=d) for d in (1e-6, 1e-5, 1e-4)]
)
REGIMES = ("R1_constant", "R2_drifting", "R3_break")
POSTHOC = ("R3b_mild_break",)
OUT = Path("experiments/results/stage4_hedge_ratio_synthetic.json")
OUT_POSTHOC = Path("experiments/results/stage4_hedge_ratio_synthetic_posthoc.json")


def simulate(regime: str, rng: np.random.Generator):
    idx = pd.bdate_range("2010-01-01", periods=N)
    lx = 4 + np.cumsum(rng.normal(3e-4, 0.015, N))
    sigma = SD_U * np.sqrt(2 * KAPPA)
    u = simulate_ou(KAPPA, 0.0, sigma, N, rng=rng)
    if regime == "R1_constant":
        beta = np.full(N, 0.8)
    elif regime == "R2_drifting":
        beta = 0.8 + np.cumsum(rng.normal(0, 0.002, N))
    else:
        beta = np.full(N, 0.8)
        beta[N_TRAIN + (N - N_TRAIN) // 2 :] = 1.05 if regime == "R3_break" else 0.85
    ly = 0.3 + beta * lx + u
    return pd.Series(ly, idx), pd.Series(lx, idx), pd.Series(beta, idx)


def git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def main() -> None:
    warnings.filterwarnings("ignore")
    posthoc = "--posthoc" in sys.argv
    regimes = POSTHOC if posthoc else REGIMES
    rng = np.random.default_rng(SEED + (1 if posthoc else 0))
    rows = []
    for regime in regimes:
        for sim in range(SIMS):
            ly, lx, beta = simulate(regime, rng)
            ry, rx = np.exp(ly).pct_change().fillna(0.0), np.exp(lx).pct_change().fillna(0.0)
            train_end = ly.index[N_TRAIN - 1]
            for spec in SPECS:
                path = hedge_path(ly, lx, spec, train_end)["beta"]
                err = (path.iloc[N_TRAIN:] - beta.iloc[N_TRAIN:]).to_numpy()
                run = run_pair(ly, lx, ry, rx, spec, Z_WINDOW, PARAMS, train_end)
                pnl = run["returns"]["pnl"].iloc[N_TRAIN:].to_numpy()
                tr = run["trades"]
                n_tr = len(tr)
                rows.append(
                    {
                        "regime": regime,
                        "sim": sim,
                        "method": spec.label,
                        "beta_rmse": float(np.sqrt(np.nanmean(err**2))),
                        "sharpe": sharpe(pnl),
                        "mean_pnl_bps": float(pnl.mean() * 1e4),
                        "turnover_per_year": float(
                            run["returns"]["trade"].iloc[N_TRAIN:].sum() / (len(pnl) / 252)
                        ),
                        "n_trades": n_tr,
                        "stop_share": float((tr["exit_reason"] == "stop").mean())
                        if n_tr
                        else np.nan,
                    }
                )
        print("done", regime, flush=True)
    df = pd.DataFrame(rows)

    def agg(g: pd.DataFrame) -> pd.Series:
        out = {}
        for c in (
            "beta_rmse",
            "sharpe",
            "mean_pnl_bps",
            "turnover_per_year",
            "n_trades",
            "stop_share",
        ):
            v = g[c].dropna()
            out[c] = float(v.mean())
            out[c + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
        return pd.Series(out)

    table = df.groupby(["regime", "method"], sort=False).apply(agg).reset_index()
    order = [s.label for s in SPECS]
    table["method"] = pd.Categorical(table["method"], order, ordered=True)
    table = table.sort_values(["regime", "method"]).reset_index(drop=True)
    pd.set_option("display.width", 220)
    print(
        table[
            [
                "regime",
                "method",
                "beta_rmse",
                "sharpe",
                "sharpe_se",
                "mean_pnl_bps",
                "turnover_per_year",
                "n_trades",
                "stop_share",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    paired = []
    static = df[df["method"] == "static"].set_index(["regime", "sim"])
    for (regime, method), g in df.groupby(["regime", "method"], sort=False):
        if method == "static":
            continue
        g = g.set_index(["regime", "sim"])
        d_sh = (g["sharpe"] - static["sharpe"]).dropna()
        d_rm = g["beta_rmse"] - static["beta_rmse"]
        paired.append(
            {
                "regime": regime,
                "method": method,
                "d_sharpe": float(d_sh.mean()),
                "d_sharpe_se": float(d_sh.std(ddof=1) / np.sqrt(len(d_sh))),
                "d_beta_rmse": float(d_rm.mean()),
                "d_beta_rmse_se": float(d_rm.std(ddof=1) / np.sqrt(len(d_rm))),
            }
        )
    print(pd.DataFrame(paired).round(3).to_string(index=False))
    ranks = {}
    for m in order:
        per_regime = []
        for regime in regimes:
            sub = table[table["regime"] == regime].sort_values("sharpe", ascending=False)
            per_regime.append(int(list(sub["method"].astype(str)).index(m)) + 1)
        ranks[m] = {"sharpe_rank_by_regime": per_regime, "worst_rank": max(per_regime)}
    out = {
        "experiment": "stage4_hedge_ratio_synthetic" + ("_posthoc" if posthoc else ""),
        "label": "POST-HOC regime, added after seeing the pre-specified run"
        if posthoc
        else "pre-specified",
        "seed": SEED,
        "git_commit": git_commit(),
        "n": N,
        "n_train": N_TRAIN,
        "sims": SIMS,
        "params": PARAMS.__dict__,
        "z_window": Z_WINDOW,
        "table": table.assign(method=table["method"].astype(str)).to_dict(orient="records"),
        "paired_vs_static": paired,
        "ranks": ranks,
    }
    dest = OUT_POSTHOC if posthoc else OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(ranks))
    print("wrote", dest)


if __name__ == "__main__":
    sys.exit(main())
