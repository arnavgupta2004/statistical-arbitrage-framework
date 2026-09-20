"""Stage 3 experiment: pair discovery on ONE training window, with an empirical null.

Pre-specified before any result was seen (nothing below was changed after running):

    Training window   2011-01-03 .. 2015-12-31  (membership reliable from 2011; early, so it is
                      research/training data under any later train/validation/holdout split)
    Universe          S&P 500 members on 2015-12-31 (point in time), identity-verified
    Screen            default ScreenConfig (alpha 0.05, half-life 5-60 trading days, beta > 0,
                      k = 5 same-sector correlation neighbours, 10 shifted-null panels), seed below
    Look out of sample  the next 252 sessions (2016), using the TRAINING hedge ratio and mean.  This
                      is a first descriptive check for Experiment A, NOT a backtest and NOT used to
                      change anything; Stage 5's walk-forward supersedes it.

Hypotheses
    H1  The screen finds more calibrated discoveries than the null expects: estimated FDR < 1.
    H2  Selected pairs' spreads stay mean-reverting out of sample more often than the baseline of
        non-significant candidates (share with a rejecting out-of-sample ADF test, and share with a
        negative out-of-sample reversion slope).
    H3  Selected spreads do not "blow up" out of sample (out-of-sample sd within ~1.5x training sd).

Out-of-sample spread: s = log y - beta * log x with the training beta, minus the TRAINING mean of
that spread.  beta is not re-estimated, so standard ADF tables (no cointegration adjustment) are the
right ones, but Stage 2 showed those are not calibrated on real data either; the comparison with the
baseline group, not the absolute rejection rate, is the evidence.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage3_pair_discovery
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from statarb.config import load_config
from statarb.data.pipeline import DataPipeline
from statarb.models.ou import fit_ou
from statarb.selection.adf import adf_test
from statarb.selection.screening import ScreenConfig, discover_pairs

SEED = 20260920
TRAIN = ("2011-01-03", "2015-12-31")
OOS_SESSIONS = 252
OUT = Path("experiments/results")


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


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


def oos_metrics(spread_full: pd.Series, n_train: int, spread_sd_train: float) -> dict:
    """Metrics of the out-of-sample spread built with training beta and training mean."""
    s_oos = spread_full.iloc[n_train:] - spread_full.iloc[:n_train].mean()
    x = s_oos.to_numpy()
    lag, lead = x[:-1], x[1:]
    b = float(np.polyfit(lag, lead, 1)[0])  # AR(1) slope; < 1 means reverting towards the mean
    try:
        adf_p = adf_test(x, "c", maxlag=10, autolag="aic").pvalue
    except ValueError:
        adf_p = float("nan")
    ou = fit_ou(x, bias_correct=True)
    return {
        "oos_sd_ratio": float(np.std(x, ddof=1) / spread_sd_train),
        "oos_mean_shift_sd": float(abs(x.mean()) / spread_sd_train),
        "oos_ar1_b": b,
        "oos_adf_p": adf_p,
        "oos_half_life": ou.half_life,
    }


def group_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    rej = int((df["oos_adf_p"] < 0.05).sum())
    rev = int((df["oos_ar1_b"] < 1.0).sum())
    return {
        "n": n,
        "share_oos_adf_rejects_5pct": rej / n if n else float("nan"),
        "ci95_adf": wilson(rej, n),
        "share_reverting_slope": rev / n if n else float("nan"),
        "median_oos_sd_ratio": float(df["oos_sd_ratio"].median()) if n else float("nan"),
        "share_sd_ratio_below_1p5": float((df["oos_sd_ratio"] < 1.5).mean()) if n else float("nan"),
        "median_oos_mean_shift_sd": float(df["oos_mean_shift_sd"].median()) if n else float("nan"),
        "median_oos_half_life": float(df["oos_half_life"].median()) if n else float("nan"),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    pipe = DataPipeline(load_config("configs/data_default.yaml"))
    cfg = ScreenConfig(seed=SEED, n_null_panels=10)
    print("discovering pairs ...", flush=True)
    res = discover_pairs(pipe, TRAIN[0], TRAIN[1], cfg)
    pairs = res.pairs
    summary = res.summary()
    print(json.dumps(summary, indent=1), flush=True)

    reasons = (
        pd.Series(res.excluded)
        .str.replace(r"[\d.]+", "#", regex=True)
        .str.replace(r"#%|#e\+#", "#", regex=True)
    )
    exclusion_counts = reasons.value_counts().to_dict()
    print("exclusions:", exclusion_counts, flush=True)

    # ---- out-of-sample look: training beta and mean, next OOS_SESSIONS sessions ----------------
    sessions = pipe.calendar.sessions(TRAIN[0], "2016-12-31")
    n_train = int((sessions <= pd.Timestamp(TRAIN[1])).sum())
    oos_end = sessions[n_train + OOS_SESSIONS - 1]
    used = sorted(set(pairs["y"]) | set(pairs["x"]))
    panel = pipe.panel(used, TRAIN[0], oos_end)  # as_of = oos_end
    logp = np.log(panel.tr_close())
    rows = []
    for r in pairs.itertuples():
        s = logp[r.y] - r.beta * logp[r.x]
        if s.isna().any():
            continue
        rows.append({"y": r.y, "x": r.x, **oos_metrics(s, n_train, r.spread_sd)})
    oos = pd.DataFrame(rows)
    merged = pairs.merge(oos, on=["y", "x"])
    sig = merged["p_calibrated"] <= cfg.alpha
    groups = {
        "selected": merged[merged["selected"]],
        "significant_not_selected": merged[sig & ~merged["selected"]],
        "not_significant_baseline": merged[~sig],
    }
    oos_summary = {k: group_summary(v) for k, v in groups.items()}
    print(json.dumps(oos_summary, indent=1), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    sel = res.selected[
        [
            "rank",
            "y",
            "x",
            "sector",
            "beta",
            "alpha_intercept",
            "T",
            "p_calibrated",
            "half_life",
            "half_life_lo",
            "half_life_hi",
            "stationary_sd",
            "edge_bps_per_day",
            "beta_rel_change",
            "adf_p_half1",
            "adf_p_half2",
            "corr",
        ]
    ].merge(oos, on=["y", "x"], how="left")
    sel.to_csv(OUT / "stage3_selected_pairs_2011_2015.csv", index=False)
    out = {
        "experiment": "stage3_pair_discovery",
        "seed": SEED,
        "git_commit": git_commit(),
        "data_fingerprint": pipe.fingerprint(),
        "train_window": TRAIN,
        "oos_window": [str(sessions[n_train].date()), str(oos_end.date())],
        "screen_config": cfg.model_dump(),
        "funnel": {
            "members_on_train_end": res.meta["n_members_train_end"],
            "n_excluded": len(res.excluded),
            "exclusion_reasons": exclusion_counts,
            **summary,
        },
        "oos_by_group": oos_summary,
        "universe_caveats": res.meta["caveats"],
        "note": "one pre-specified window; no parameter was changed after seeing results",
    }
    (OUT / "stage3_pair_discovery.json").write_text(json.dumps(out, indent=2, default=str))
    print("wrote", OUT / "stage3_pair_discovery.json")
    pipe.close()


if __name__ == "__main__":
    sys.exit(main())
