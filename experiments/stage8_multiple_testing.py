"""Stage 8 experiment: how much of the apparent gross performance is selection luck?

Fixed before it was run.  Everything is computed from series and screens that already exist for the
RESEARCH phase (2015-2018); validation (2019-2021) and the holdout (2022+) are not touched, and this
stage adds no trial (it is logged as a diagnostic).

Inputs
    Strategy series  the daily gross (``pnl``) and net (``net``) returns of the 15 real-data strategy
                     trials that share the 2015-2018 sample: the 9 Stage 5 pair configurations
                     (``stage5_daily_returns.csv``, ``stage6_daily_net_returns.csv``) and the 6 Stage 7
                     PCA configurations (``stage7_daily_returns.csv``).  The Stage 7 no-factor control is
                     a diagnostic, not a trial, and is left out.
    Trial count      ``Registry.count_trials("strategy")``: 24 distinct real-data strategy trials (the 15
                     above, the Stage 3 screen, and Stage 4's 8 variants on 2016 only).  24 is the
                     *primary* N for deflation.  It is a lower bound: design choices made while building
                     (z-window, k grid, which hedge methods to carry) were not registered.
    Pair screens     the calibrated p-values of every candidate pair of the four fold screens and of the
                     Stage 3 window, recomputed with the seeds and configurations of Stages 3 and 5.

Protocol (every choice fixed here)
    Deflated Sharpe  daily Sharpe, skewness and kurtosis of each series; ``V[SR]`` = variance of the 15
                     trials' daily Sharpe estimates; ``N`` = 24 (primary), the effective number of
                     independent trials from the correlation matrix (two estimates) and 15 (the series
                     available) as sensitivities.  DSR >= 0.95 is the threshold.
    Family tests     White's Reality Check and Hansen's SPA (consistent p-value primary) for the null "no
                     strategy in the family has a positive mean return", stationary bootstrap of the dates
                     (mean block 10 days; 5 and 20 as sensitivities), 5,000 resamples; families: all 15,
                     the 9 pair configurations, the 6 PCA configurations.  Romano-Wolf step-down adjusted
                     p-values per strategy; BH and BY on the raw bootstrap p-values (q = 0.05).
    FDR on screens   Benjamini-Hochberg and -Yekutieli on the calibrated p-values of each screen, q in
                     {0.05, 0.10, 0.20}: the FDR-controlled count of "cointegrated" pairs.
    PBO              CSCV with 16 blocks (12,870 splits; 8 blocks as a sensitivity) over the 15 gross
                     series, and within each family.
    Decision rule    a positive gross result *survives* Stage 8 only if the best strategy has DSR >= 0.95 at
                     N = 24 **and** the SPA p-value of its family is <= 0.05.  Otherwise it is declared
                     not distinguishable from selection luck.  Net series are tested too, for completeness.

Post-hoc addition (labelled).  After the first run it was clear that BH on the screens is resolution-limited:
with 10 null panels the smallest attainable calibrated p is ~1e-4, while BH at q = 0.05 with ~1,300
candidates needs the first rejection at p <= 4e-5, so "zero discoveries" is partly forced by the null's
resolution, not by the data.  Storey's pi0 (share of true nulls, from the p-values above lambda) and the p-value
histogram do not depend on that resolution and were added, then the experiment re-run (registry run 2; the
strategy-series results are unchanged, being deterministic).

Hypotheses (priors written before running)
    H8a  No strategy has DSR >= 0.95 at N = 24 (gross).  Prior: yes.  The best gross Sharpe (~1.1) is close
         to what the best of 24 luck-only strategies would show at this sample length.
    H8b  The Reality Check and SPA do not reject at 5 % for the 15 gross series.  Prior: probably (~65 %);
         the single-strategy p of the best is ~0.01 and the family is partly correlated, so this is close.
    H8c  No strategy has a Romano-Wolf adjusted p <= 0.05.  Prior: same as H8b.
    H8d  BH and BY find zero cointegrated pairs at q = 0.05 in every screen (estimated FDR was 0.90-1.00).
    H8e  (descriptive) PBO of the PCA family is not below 0.3: within a family that differs only in k and the
         signal definition, the in-sample winner is not reliably the out-of-sample winner.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage8_multiple_testing
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
from statarb.research.registry import Registry
from statarb.selection.screening import ScreenConfig, discover_pairs
from statarb.statistics.multiple_testing import (
    benjamini_hochberg,
    benjamini_yekutieli,
    cscv_pbo,
    reality_check,
    romano_wolf,
    spa_test,
)
from statarb.statistics.sharpe import (
    deflated_sharpe,
    effective_trials,
    expected_max_sharpe,
    min_backtest_years,
    min_track_record_length,
    sharpe_moments,
)

R = Path("experiments/results")
N_BOOT = 5000
MEAN_BLOCK = 10.0
SEED = 20260924
PERIODS = 252
N_TRIALS_PRIMARY = 24
STAGE3_SEED, STAGE5_SEED = 20260920, 20260922


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def load_series() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    s5 = pd.read_csv(R / "stage5_daily_returns.csv", parse_dates=["date"])
    s6 = pd.read_csv(R / "stage6_daily_net_returns.csv", parse_dates=["date"])
    s7 = pd.read_csv(R / "stage7_daily_returns.csv", parse_dates=["date"])
    grid7 = json.loads((R / "stage7_pca.json").read_text())["grid"]
    s7 = s7[s7["config"].isin(grid7)]
    pairs_cfg = sorted(s5["config"].unique())
    gross = pd.concat(
        [
            s5.pivot(index="date", columns="config", values="pnl")[pairs_cfg],
            s7.pivot(index="date", columns="config", values="pnl")[grid7],
        ],
        axis=1,
    )
    net = pd.concat(
        [
            s6.pivot(index="date", columns="config", values="net")[pairs_cfg],
            s7.pivot(index="date", columns="config", values="net")[grid7],
        ],
        axis=1,
    )
    if not gross.index.equals(net.index) or gross.isna().any().any() or net.isna().any().any():
        raise ValueError("strategy series do not share one complete date index")
    return gross, net, pairs_cfg, grid7


def sharpe_table(x: pd.DataFrame, n_primary: int, n_effs: dict, var_sr: float) -> dict:
    out = {}
    for c in x.columns:
        sr, sk, ku, n = sharpe_moments(x[c].to_numpy())
        row = {
            "sharpe_annual": sr * np.sqrt(PERIODS),
            "skew": sk,
            "kurtosis": ku,
            "n_days": n,
            "psr_vs_zero": deflated_sharpe(sr, n, sk, ku, 1, var_sr)["psr_vs_zero"],
            "min_track_record_years_vs_zero": min_track_record_length(sr, 0.0, sk, ku) / PERIODS,
        }
        for label, n_trials in {"n24": n_primary, **n_effs}.items():
            d = deflated_sharpe(sr, n, sk, ku, n_trials, var_sr)
            row[f"dsr_{label}"] = d["dsr"]
            row[f"sr0_annual_{label}"] = d["sr0"] * np.sqrt(PERIODS)
        out[c] = row
    return out


def family_tests(x: pd.DataFrame, block: float = MEAN_BLOCK) -> dict:
    d = x.to_numpy()
    rc = reality_check(d, N_BOOT, block, SEED)
    spa = spa_test(d, N_BOOT, block, SEED)
    rw = romano_wolf(d, N_BOOT, block, SEED)
    bh_rej, bh_adj = benjamini_hochberg(rw["p_raw"], 0.05)
    by_rej, by_adj = benjamini_yekutieli(rw["p_raw"], 0.05)
    return {
        "n_strategies": d.shape[1],
        "best": x.columns[rc["best"]],
        "reality_check_p": rc["p_value"],
        "spa_p": {k: spa[k] for k in ("lower", "consistent", "upper")},
        "spa_n_poor": spa["n_poor"],
        "per_strategy": {
            c: {
                "t": float(rw["t"][j]),
                "p_raw": float(rw["p_raw"][j]),
                "p_romano_wolf": float(rw["p_adjusted"][j]),
                "p_bh": float(bh_adj[j]),
                "p_by": float(by_adj[j]),
            }
            for j, c in enumerate(x.columns)
        },
        "n_rejected_romano_wolf_5pct": int((rw["p_adjusted"] <= 0.05).sum()),
        "n_rejected_bh_5pct": int(bh_rej.sum()),
        "n_rejected_by_5pct": int(by_rej.sum()),
    }


def screen_fdr(pipe, label, train_start, train_end, cfg) -> dict:
    res = discover_pairs(pipe, train_start, train_end, cfg)
    p = res.pairs["p_calibrated"].dropna().to_numpy()
    out = {
        "window": [str(pd.Timestamp(train_start).date()), str(pd.Timestamp(train_end).date())],
        "n_candidates": len(res.pairs),
        "n_p_values": len(p),
        "calibrated_discoveries_at_5pct_uncorrected": int((p <= 0.05).sum()),
        "expected_under_null_at_5pct": 0.05 * len(p),
        "smallest_p": float(p.min()),
    }
    m = len(p)
    out["storey_pi0"] = {
        str(lam): float(min(1.0, (p > lam).sum() / (m * (1 - lam)))) for lam in (0.3, 0.5, 0.7)
    }
    out["p_histogram_deciles"] = np.histogram(p, bins=10, range=(0, 1))[0].tolist()
    out["attainable_p_floor"] = 1.0 / (sum(len(a) for a in res.null_stats) + 1)
    for q in (0.05, 0.10, 0.20):
        out[f"bh_q{q:.2f}"] = int(benjamini_hochberg(p, q)[0].sum())
        out[f"by_q{q:.2f}"] = int(benjamini_yekutieli(p, q)[0].sum())
    print(
        f"  screen {label}: {len(p)} p-values, uncorrected@5% {out['calibrated_discoveries_at_5pct_uncorrected']} "
        f"(null expects {out['expected_under_null_at_5pct']:.0f}), BH q=.05 {out['bh_q0.05']}, "
        f"BH q=.20 {out['bh_q0.20']}, BY q=.20 {out['by_q0.20']}, pi0(0.5) {out['storey_pi0']['0.5']:.2f}, "
        f"p floor {out['attainable_p_floor']:.1e}",
        flush=True,
    )
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    gross, net, pairs_cfg, pca_cfg = load_series()
    registry = Registry()
    n_registry = registry.count_trials("strategy")["distinct_specs"]
    assert n_registry == N_TRIALS_PRIMARY, f"registry has {n_registry} strategy trials, not 24"
    n_days = len(gross)

    eff = effective_trials(gross.to_numpy())
    n_effs = {
        "neff_corr": eff["n_eff_average_correlation"],
        "neff_part": eff["n_eff_participation_ratio"],
        "n15": 15,
    }
    var_gross = float(np.var([sharpe_moments(gross[c].to_numpy())[0] for c in gross], ddof=1))
    var_net = float(np.var([sharpe_moments(net[c].to_numpy())[0] for c in net], ddof=1))

    print("Sharpe deflation ...", flush=True)
    out = {
        "experiment": "stage8_multiple_testing",
        "seed": SEED,
        "git_commit": git_commit(),
        "n_days": n_days,
        "registry_strategy_trials": {
            "distinct": n_registry,
            "runs": registry.count_trials("strategy")["total_runs"],
        },
        "effective_trials_gross": eff,
        "var_daily_sharpe": {"gross": var_gross, "net": var_net},
        "gross": sharpe_table(gross, N_TRIALS_PRIMARY, n_effs, var_gross),
        "net": sharpe_table(net, N_TRIALS_PRIMARY, n_effs, var_net),
    }
    years = n_days / PERIODS
    out["selection_luck_table"] = {
        "sample_years": years,
        "expected_max_annual_sharpe_of_n_luck_only_strategies": {
            str(n): expected_max_sharpe(n, 1.0 / (n_days - 1)) * np.sqrt(PERIODS)
            for n in (1, 5, 15, 24, 50, 100)
        },
        "min_backtest_years_for_annual_sharpe_1": {
            str(n): min_backtest_years(n, 1.0) for n in (1, 5, 15, 24, 50, 100)
        },
        "sr0_annual_at_n24_from_trial_dispersion": expected_max_sharpe(24, var_gross)
        * np.sqrt(PERIODS),
    }

    print("Family tests ...", flush=True)
    fams = {"all_15": list(gross.columns), "pairs_9": pairs_cfg, "pca_6": pca_cfg}
    out["family_gross"] = {k: family_tests(gross[v]) for k, v in fams.items()}
    out["family_net"] = {k: family_tests(net[v]) for k, v in fams.items()}
    sens = {}
    for b in (5.0, 20.0):
        ft = family_tests(gross, b)
        sens[str(b)] = {
            "reality_check_p": ft["reality_check_p"],
            "spa_consistent_p": ft["spa_p"]["consistent"],
        }
    out["family_gross_block_sensitivity"] = sens

    print("Probability of backtest overfitting ...", flush=True)
    out["pbo_gross"] = {
        k: {
            "blocks16": cscv_pbo(gross[v].to_numpy(), 16),
            "blocks8": cscv_pbo(gross[v].to_numpy(), 8),
        }
        for k, v in fams.items()
    }

    print("FDR on the pair screens (re-running the screens; a few minutes) ...", flush=True)
    pipe = DataPipeline(load_config("configs/data_default.yaml"))
    from statarb.backtest.walkforward import make_folds

    split = pipe.cfg.split
    screens = {
        "stage3_2011_2015": screen_fdr(
            pipe,
            "stage3_2011_2015",
            "2011-01-03",
            "2015-12-31",
            ScreenConfig(seed=STAGE3_SEED, n_null_panels=10),
        )
    }
    for f in make_folds(split, pipe.calendar, 2015, 2018, 4, "rolling"):
        screens[f"fold_test_{f.test_start.year}"] = screen_fdr(
            pipe,
            f"fold {f.test_start.year}",
            f.train_start,
            f.train_end,
            ScreenConfig(seed=STAGE5_SEED, n_null_panels=10),
        )
    out["screen_fdr"] = screens
    fingerprint = pipe.fingerprint()
    pipe.close()
    out["data_fingerprint"] = fingerprint

    # the pre-specified decision rule
    best_g = max(gross.columns, key=lambda c: out["gross"][c]["sharpe_annual"])
    best_family = "pca_6" if best_g in pca_cfg else "pairs_9"
    dsr_best = out["gross"][best_g]["dsr_n24"]
    spa_best = out["family_gross"][best_family]["spa_p"]["consistent"]
    out["decision"] = {
        "rule": "survives only if DSR(N=24) >= 0.95 and SPA(consistent, own family) <= 0.05",
        "best_gross_strategy": best_g,
        "best_gross_sharpe_annual": out["gross"][best_g]["sharpe_annual"],
        "dsr_n24": dsr_best,
        "spa_consistent_p_own_family": spa_best,
        "survives": bool(dsr_best >= 0.95 and spa_best <= 0.05),
        "any_gross_strategy_dsr_ge_95": [
            c for c in gross.columns if out["gross"][c]["dsr_n24"] >= 0.95
        ],
        "any_net_strategy_positive_sharpe": [
            c for c in net.columns if out["net"][c]["sharpe_annual"] > 0
        ],
    }

    R.mkdir(parents=True, exist_ok=True)
    (R / "stage8_multiple_testing.json").write_text(json.dumps(out, indent=2, default=float))
    registry.register(
        stage=8,
        kind="diagnostic",
        name="multiple_testing_correction_research_phase",
        phases=["research"],
        windows={"test_years": [2015, 2018]},
        universe="the 15 real-data strategy trials sharing the 2015-2018 sample",
        strategy="none (Deflated Sharpe, Reality Check, SPA, Romano-Wolf, BH/BY, CSCV-PBO)",
        parameters={
            "n_boot": N_BOOT,
            "mean_block": MEAN_BLOCK,
            "n_trials_primary": N_TRIALS_PRIMARY,
        },
        results={
            "best_gross": best_g,
            "dsr_n24": dsr_best,
            "spa_p": spa_best,
            "survives": out["decision"]["survives"],
        },
        seed=SEED,
        data_fingerprint=fingerprint,
        git_commit=git_commit(),
        notes="adds no trial; research phase only; validation and holdout untouched",
    )

    print(f"\nbest gross strategy: {best_g} (Sharpe {out['gross'][best_g]['sharpe_annual']:+.2f})")
    print(
        f"  PSR vs 0 {out['gross'][best_g]['psr_vs_zero']:.3f}; DSR N=24 {dsr_best:.3f}; SR0 {out['gross'][best_g]['sr0_annual_n24']:.2f}"
    )
    for k, v in out["family_gross"].items():
        print(
            f"  family {k}: RC p {v['reality_check_p']:.3f}, SPA(c) {v['spa_p']['consistent']:.3f}, RW rejects {v['n_rejected_romano_wolf_5pct']}"
        )
    print("decision:", out["decision"]["survives"])
    print("wrote", R / "stage8_multiple_testing.json")


if __name__ == "__main__":
    sys.exit(main())
