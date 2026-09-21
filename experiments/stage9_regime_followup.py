"""Stage 9 post-hoc follow-up: is the dispersion-regime effect more than an artefact of persistent labels?

Written AFTER experiment B (``stage9_regimes_breaks``), which found that the three PCA reversal strategies
have a far higher gross Sharpe in the high-dispersion regime and that these survive BH at 10 % across the 45
pre-specified tests (contradicting my prior H9e).  This script probes that one finding; it is post-hoc, it does
not replace the pre-specified verdict, and any further look at these regimes would count as another trial.

Concern.  Regimes last months, but the bootstrap of experiment B resamples 10-day blocks, so it can understate
the sampling variability of a regime contrast (few independent episodes).  Checks:
  1  the bootstrap p-value under longer blocks (5, 10, 21, 63 days);
  2  how persistent the label is (number and length of episodes);
  3  the contrast year by year (is it one episode?);
  4  a circular-SHIFT test that keeps the label's persistence and its overall pattern exactly and destroys only its
     alignment with the returns: the strategy's Sharpe difference between the shifted labels' two states, over every
     shift of at least 63 days; p = share of shifts with an |difference| at least as large.  Validated on pure noise
     inside this script (rejection rate must be within 1-12 % at the 5 % level).

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage9_regime_followup
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

from experiments.stage8_multiple_testing import load_series
from statarb.backtest.pca_walkforward import prepare_pca_fold
from statarb.backtest.walkforward import make_folds
from statarb.config import load_config
from statarb.data.pipeline import DataPipeline
from statarb.research.registry import Registry
from statarb.selection.screening import ScreenConfig
from statarb.statistics.multiple_testing import benjamini_hochberg
from statarb.statistics.regimes import causal_regimes, regime_sharpe_difference

R = Path("experiments/results")
SEED = 20260927
MIN_SHIFT = 63


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def _sharpe_diff(r: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Sharpe(True state) - Sharpe(False state) for each row of ``lab`` (S x n) against one series."""
    out = np.empty(len(lab))
    for i, m in enumerate(lab):
        a, b = r[m], r[~m]
        out[i] = a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)
    return out * np.sqrt(252)


def shift_test(r: np.ndarray, labels: np.ndarray, min_shift: int = MIN_SHIFT) -> dict:
    lab = labels.astype(bool)
    n = len(r)
    shifts = np.arange(min_shift, n - min_shift + 1)
    obs = _sharpe_diff(r, lab[None, :])[0]
    null = _sharpe_diff(r, np.array([np.roll(lab, s) for s in shifts]))
    return {
        "difference": float(obs),
        "p_value": float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null))),
        "n_shifts": len(shifts),
    }


def _ann_sharpe(x: np.ndarray) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 5 else float("nan")


def episodes(lab: np.ndarray) -> dict:
    change = np.flatnonzero(np.diff(lab.astype(int)) != 0) + 1
    runs = np.diff(np.r_[0, change, len(lab)])
    first = lab[0]
    trues = runs[0::2] if first else runs[1::2]
    falses = runs[1::2] if first else runs[0::2]
    return {
        "n_true_episodes": len(trues),
        "median_true_length": float(np.median(trues)),
        "n_false_episodes": len(falses),
        "median_false_length": float(np.median(falses)),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    gross, _, _, pca_cfg = load_series()
    # self-check of the shift test on pure noise with persistent labels of the same kind
    rng = np.random.default_rng(SEED)
    persistent = np.sin(np.arange(len(gross)) / 40.0) > 0
    rej = np.mean(
        [
            shift_test(
                rng.normal(0, 0.005, len(gross)), np.roll(persistent, int(rng.integers(len(gross))))
            )["p_value"]
            < 0.05
            for _ in range(150)
        ]
    )
    assert 0.01 <= rej <= 0.12, f"shift test is mis-sized on noise: {rej}"

    pipe = DataPipeline(load_config("configs/data_default.yaml"))
    folds = make_folds(pipe.cfg.split, pipe.calendar, 2015, 2018, 4, "rolling")
    fd0 = prepare_pca_fold(pipe, folds[0], ScreenConfig())
    panel = pipe.panel(list(fd0.tickers), "2011-01-03", "2018-12-31")
    reg = causal_regimes(
        panel.ret.mean(axis=1, skipna=True), panel.ret.std(axis=1), 21, 126, 252
    ).reindex(gross.index)
    pipe.close()
    lab = reg["disp_high"].to_numpy()
    assert np.isfinite(lab).all()

    out = {
        "experiment": "stage9_regime_followup",
        "posthoc": True,
        "seed": SEED,
        "git_commit": git_commit(),
        "shift_test_size_on_noise": float(rej),
        "episodes_disp_high": episodes(lab.astype(bool)),
    }
    out["share_disp_high_by_year"] = {
        str(y): float(g.mean()) for y, g in reg["disp_high"].groupby(reg.index.year)
    }
    # 1: block length
    blocks = {}
    for b in (5.0, 10.0, 21.0, 63.0):
        blocks[str(b)] = {
            c: regime_sharpe_difference(gross[c].to_numpy(), lab, 5000, b, SEED)["p_value"]
            for c in gross.columns
        }
    out["bootstrap_p_by_mean_block"] = {
        b: {
            "reversal_k5": v["pca_reversal_k5"],
            "reversal_k10": v["pca_reversal_k10"],
            "reversal_k15": v["pca_reversal_k15"],
            "n_of_15_below_5pct": int(sum(x < 0.05 for x in v.values())),
        }
        for b, v in blocks.items()
    }
    # 3: year by year, PCA strategies
    by_year = {}
    for c in pca_cfg:
        by_year[c] = {}
        for y, g in gross[c].groupby(gross.index.year):
            m = lab[gross.index.year == y].astype(bool)
            sh = _ann_sharpe
            by_year[c][str(y)] = {
                "sharpe_high": sh(g.to_numpy()[m]),
                "sharpe_low": sh(g.to_numpy()[~m]),
                "n_high": int(m.sum()),
                "n_low": int((~m).sum()),
            }
    out["by_year_pca"] = by_year
    # 4: shift test for all 15
    rows = {c: shift_test(gross[c].to_numpy(), lab) for c in gross.columns}
    rej_bh, adj = benjamini_hochberg(np.array([v["p_value"] for v in rows.values()]), 0.10)
    for v, a in zip(rows.values(), adj, strict=True):
        v["p_bh_over_15"] = float(a)
    out["shift_test"] = rows
    out["shift_test_summary"] = {
        "n_p_below_5pct": int(sum(v["p_value"] < 0.05 for v in rows.values())),
        "n_bh_rejections_at_10pct": int(rej_bh.sum()),
        "all_15_differences_positive": bool(all(v["difference"] > 0 for v in rows.values())),
    }
    (R / "stage9_regime_followup.json").write_text(json.dumps(out, indent=2, default=float))
    Registry().register(
        stage=9,
        kind="diagnostic",
        name="dispersion_regime_followup_posthoc",
        phases=["research"],
        strategy="none (post-hoc probe of one regime finding)",
        parameters={"blocks": [5, 10, 21, 63], "min_shift": MIN_SHIFT},
        seed=SEED,
        git_commit=git_commit(),
        notes="post-hoc, after the pre-specified regime tests; adds no trial; research phase only",
    )
    print("episodes:", out["episodes_disp_high"])
    print("bootstrap p by block:", json.dumps(out["bootstrap_p_by_mean_block"]))
    print(
        "shift test:",
        out["shift_test_summary"],
        {c: round(v["p_value"], 4) for c, v in rows.items() if c.startswith("pca_reversal")},
    )


if __name__ == "__main__":
    sys.exit(main())
