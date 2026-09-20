"""POST-HOC diagnostic of the Stage 3 result (added AFTER seeing it; not pre-specified).

Observation that prompted it: real candidates are calibrated-significant at 3.4 % (43 / 1259) while
the shifted null rejects 5 % by construction.  Real candidates are highly correlated sector peers;
null candidates are chance-correlated.  If the Engle-Granger statistic depends on the return
correlation of the pair, the independent-shift null is a *conservative* benchmark for correlated
pairs, and "no excess over the null" would partly reflect that rather than an absence of
cointegration.

This script re-runs the identical, seeded screen (same window, same config) and reports
  * quantiles of T for real vs null candidates;
  * the calibrated rejection rate of real candidates within return-correlation terciles;
  * how correlated the null's chosen candidates are, versus the real ones.
It is explanatory only: no parameter of the screen is changed on the basis of it.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage3_null_diagnostics
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.stage3_pair_discovery import SEED, TRAIN
from statarb.config import load_config
from statarb.data.pipeline import DataPipeline
from statarb.selection.correlation import return_correlation, top_k_neighbours
from statarb.selection.screening import ScreenConfig, discover_pairs, eligible_tickers, shift_panel

OUT = Path("experiments/results/stage3_null_diagnostics.json")


def main() -> None:
    warnings.filterwarnings("ignore")
    pipe = DataPipeline(load_config("configs/data_default.yaml"))
    cfg = ScreenConfig(seed=SEED, n_null_panels=10)
    res = discover_pairs(pipe, TRAIN[0], TRAIN[1], cfg)
    pairs = res.pairs
    pool = np.concatenate(res.null_stats)
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95]
    t_quant = {
        "quantiles": qs,
        "real_candidates_T": [float(np.quantile(pairs["T"], q)) for q in qs],
        "null_candidates_T": [float(np.quantile(pool, q)) for q in qs],
    }

    tercile = pd.qcut(pairs["corr"], 3, labels=["low", "mid", "high"])
    by_corr = {
        str(t): {
            "n": int((tercile == t).sum()),
            "corr_range": [
                float(pairs.loc[tercile == t, "corr"].min()),
                float(pairs.loc[tercile == t, "corr"].max()),
            ],
            "calibrated_rate_5pct": float((pairs.loc[tercile == t, "p_calibrated"] <= 0.05).mean()),
            "median_T": float(pairs.loc[tercile == t, "T"].median()),
        }
        for t in ["low", "mid", "high"]
    }

    panel, tickers, _ = eligible_tickers(pipe, TRAIN[0], TRAIN[1], cfg)
    logp = np.log(panel.tr_close()[tickers])
    sectors = pipe.universe().sector_of(tickers)
    rng = np.random.default_rng(SEED + 7)
    null_corr = []
    for _ in range(3):
        sh = shift_panel(logp, rng)
        c = return_correlation(sh)
        for a, b in top_k_neighbours(c, sectors, cfg.k_neighbours):
            null_corr.append(c.at[a, b])
    out = {
        "label": "POST-HOC explanatory diagnostic; the screen was not changed on its basis",
        "T_distribution": t_quant,
        "real_by_correlation_tercile": by_corr,
        "median_corr_real_candidates": float(pairs["corr"].median()),
        "median_corr_null_candidates": float(np.median(null_corr)),
        "n_real": int(len(pairs)),
        "n_null_pool": int(len(pool)),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    pipe.close()


if __name__ == "__main__":
    sys.exit(main())
