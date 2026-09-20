"""Stage 4 experiment: the strategy chain on the FROZEN Stage 3 pairs, out of sample (2016). Gross.

Descriptive, fully pre-specified, all variants reported.  Nothing here selects a hedge method or a
parameter: the synthetic study and Stage 5's walk-forward do that.  It exists to (a) show what each
hedge-ratio variant does to hedge-ratio stability and turnover on real prices, and (b) put a first
frictionless P&L number next to the Stage 3 out-of-sample statistics, with honest uncertainty.

Fixed before running
    Pairs      the 20 pairs Stage 3 selected on 2011-2015 (orientation and training beta as found),
               and, as a baseline, all 1,216 non-significant candidates of the same screen.
               Screen: default ScreenConfig, seed 20260920 (deterministic, so it is re-run here).
    Window     trade the 252 sessions of 2016; the hedge ratio is fitted on data <= 2015-12-31.
    Strategy   entry 2.0, exit 0.5, stop 4.0, max hold 60, unit sizing, daily re-hedging, no costs.
    Variants   static (z-window 30 / 60 / 120), expanding (z 60), rolling 60 / 120 / 250 (z 60),
               kalman delta 1e-5 (z 60; the middle of the synthetic grid).
    Evaluation equal-weight average of the pairs' daily P&L; Sharpe with a stationary-bootstrap 95 %
               interval (block 10, 2000 draws); trades, turnover, stop-outs; hedge-ratio drift.
    Data hygiene  a pair is dropped if either leg has a non-ordinary distribution (> 10 % of price)
               inside the trading window: vendor total returns around a spin-off are not trustworthy
               (Stage 1, section 1.8).  The number dropped is reported.

Hypothesis (H4a): if selection carried information, the selected pairs' gross Sharpe exceeds the
baseline's.  Given Stage 3's out-of-sample result (no persistence) the prior is that it does not.

Two changes were made AFTER seeing the first run and are labelled post-hoc in the output:
  * SIZE-MATCHED COMPARISON.  The baseline is an equal-weight portfolio of ~1,190 pairs and the
    selected group has ~19, so the baseline's Sharpe is inflated by diversification alone.  The fair
    comparison is the selected group's Sharpe against the distribution of Sharpes of random subsets
    of the baseline of the SAME size (5,000 draws); its percentile in that distribution is reported.
  * The hedge-ratio drift is summarised by the MEDIAN across pairs: the mean was dominated by a few
    pairs with a training beta near zero (a relative change of an almost-zero number).

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage4_real_pairs
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.stage3_pair_discovery import SEED as STAGE3_SEED
from experiments.stage3_pair_discovery import TRAIN
from statarb.backtest.pair_pnl import run_pair
from statarb.config import load_config
from statarb.data.pipeline import DataPipeline
from statarb.models.hedge_ratio import HedgeSpec
from statarb.selection.screening import ScreenConfig, discover_pairs
from statarb.signals.pairs import PairParams
from statarb.statistics.bootstrap import bootstrap_ci, sharpe

OOS_SESSIONS = 252
PARAMS = PairParams(2.0, 0.5, 4.0, 60)
VARIANTS = [
    ("static_z30", HedgeSpec("static"), 30),
    ("static_z60", HedgeSpec("static"), 60),
    ("static_z120", HedgeSpec("static"), 120),
    ("expanding_z60", HedgeSpec("expanding"), 60),
    ("rolling60_z60", HedgeSpec("rolling", window=60), 60),
    ("rolling120_z60", HedgeSpec("rolling", window=120), 60),
    ("rolling250_z60", HedgeSpec("rolling", window=250), 60),
    ("kalman1e-05_z60", HedgeSpec("kalman", delta=1e-5), 60),
]
OUT = Path("experiments/results/stage4_real_pairs.json")


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
    pipe = DataPipeline(load_config("configs/data_default.yaml"))
    res = discover_pairs(pipe, TRAIN[0], TRAIN[1], ScreenConfig(seed=STAGE3_SEED, n_null_panels=10))
    pairs = res.pairs
    groups = {
        "selected": pairs[pairs["selected"]],
        "baseline_non_significant": pairs[pairs["p_calibrated"] > 0.05],
    }
    sessions = pipe.calendar.sessions(TRAIN[0], "2016-12-31")
    n_train = int((sessions <= pd.Timestamp(TRAIN[1])).sum())
    oos_end = sessions[n_train + OOS_SESSIONS - 1]
    used = sorted(set(pairs["y"]) | set(pairs["x"]))
    panel = pipe.panel(used, TRAIN[0], oos_end)  # as_of = oos_end
    logp = np.log(panel.tr_close())
    ret = panel.ret.fillna(0.0)
    div_yield = (panel.dividends / panel.close.shift(1)).loc[sessions[n_train] : oos_end].max()
    bad = set(div_yield.index[div_yield > 0.10])
    train_end = pd.Timestamp(TRAIN[1])
    print(
        f"{len(pairs)} candidates; legs with a non-ordinary distribution in 2016: {sorted(bad)}",
        flush=True,
    )

    results: dict = {}
    mats: dict = {}
    for gname, g in groups.items():
        keep = [r for r in g.itertuples() if r.y not in bad and r.x not in bad]
        dropped = len(g) - len(keep)
        print(
            f"{gname}: {len(keep)} pairs ({dropped} dropped for non-ordinary distributions)",
            flush=True,
        )
        for label, spec, zwin in VARIANTS:
            pnl_cols, turn, ntr, stops, wins, bdrift = [], [], [], [], [], []
            for r in keep:
                ly, lx = logp[r.y], logp[r.x]
                run = run_pair(ly, lx, ret[r.y], ret[r.x], spec, zwin, PARAMS, train_end)
                oos = run["returns"].iloc[n_train:]
                pnl_cols.append(oos["pnl"].to_numpy())
                turn.append(float(oos["trade"].sum()))
                tr = run["trades"]
                ntr.append(len(tr))
                stops.append(int((tr["exit_reason"] == "stop").sum()))
                wins.append(int((tr["pnl"] > 0).sum()))
                b = run["beta"].iloc[n_train:]
                bdrift.append(
                    float((b - b.iloc[0]).abs().mean() / abs(r.beta)) if b.notna().all() else np.nan
                )
            mat = np.vstack(pnl_cols)
            mats[(gname, label)] = mat
            port = mat.mean(axis=0)  # equal-weight average of pair P&L (pair capital = 1)
            sh, ci, _ = bootstrap_ci(port, sharpe, n_boot=2000, mean_block=10, seed=1)
            t_total = int(np.sum(ntr))
            results.setdefault(gname, {})[label] = {
                "n_pairs": len(keep),
                "n_dropped": dropped,
                "sharpe": sh,
                "sharpe_ci95": list(ci),
                "mean_daily_bps": float(port.mean() * 1e4),
                "total_return_pct_per_unit": float(port.sum() * 100),
                "trades_per_pair": t_total / len(keep),
                "stop_share": float(np.sum(stops) / t_total) if t_total else np.nan,
                "win_rate": float(np.sum(wins) / t_total) if t_total else np.nan,
                "turnover_per_pair": float(np.mean(turn)),
                "hedge_ratio_drift_rel_median": float(np.nanmedian(bdrift)),
                "pairs_with_no_trade": int(np.sum(np.array(ntr) == 0)),
            }
            r_ = results[gname][label]
            print(
                f"  {gname:24s} {label:16s} Sharpe {r_['sharpe']:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] "
                f"trades/pair {r_['trades_per_pair']:.1f} turnover {r_['turnover_per_pair']:.1f} "
                f"beta drift(med) {r_['hedge_ratio_drift_rel_median']:.3f}",
                flush=True,
            )
    # ---- POST-HOC: selected vs random baseline subsets of the same size ------------------------
    rng = np.random.default_rng(11)
    matched = {}
    for label, _, _ in VARIANTS:
        sel, base = mats[("selected", label)], mats[("baseline_non_significant", label)]
        k = sel.shape[0]
        draws = np.array(
            [
                sharpe(base[rng.choice(base.shape[0], k, replace=False)].mean(axis=0))
                for _ in range(5000)
            ]
        )
        s_sel = sharpe(sel.mean(axis=0))
        matched[label] = {
            "k": k,
            "selected_sharpe": s_sel,
            "random_subset_median": float(np.nanmedian(draws)),
            "random_subset_5_95": [
                float(np.nanquantile(draws, 0.05)),
                float(np.nanquantile(draws, 0.95)),
            ],
            "selected_percentile": float((draws < s_sel).mean()),
        }
        m = matched[label]
        print(
            f"  size-matched {label:16s} selected {s_sel:+.2f} | random {k}-pair subsets median "
            f"{m['random_subset_median']:+.2f} (5-95%: {m['random_subset_5_95'][0]:+.2f}, "
            f"{m['random_subset_5_95'][1]:+.2f}) | pctile {m['selected_percentile']:.0%}",
            flush=True,
        )
    results["size_matched_post_hoc"] = matched
    out = {
        "experiment": "stage4_real_pairs",
        "git_commit": git_commit(),
        "data_fingerprint": pipe.fingerprint(),
        "train_end": TRAIN[1],
        "oos_window": [str(sessions[n_train].date()), str(oos_end.date())],
        "params": PARAMS.__dict__,
        "screen_seed": STAGE3_SEED,
        "legs_dropped_for_distributions": sorted(bad),
        "results": results,
        "note": "descriptive, gross of costs, pre-specified variants all reported",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)
    pipe.close()


if __name__ == "__main__":
    sys.exit(main())
