"""Stage 9 experiment B: market regimes, structural breaks and the stability of the PCA factor space.

Fixed before it was run.  Research phase only (2015-2018): validation and the holdout are not touched; no trial
is added (diagnostics, registered as such).

Series   the 15 real-data strategy trials sharing the 2015-2018 sample (9 pair configurations, 6 PCA
         configurations), gross daily returns for the tests and net returns for the regime table.
Market   the equal-weight return of the fold-0 eligible universe (the names identified as of 2014-12-31), fetched from
         2011 so that every indicator has history; cross-sectional dispersion = the daily standard deviation of
         those names' returns.  A survivor-tilted proxy (Section 2) used only to *classify* days.

Regimes (fixed in advance; label for day t uses data through t-1, threshold = expanding median, causal)
    vol_high    21-day realised market volatility above its expanding median
    trend_up    126-day compounded market return above zero
    disp_high   21-day mean cross-sectional dispersion above its expanding median
Each of the 15 gross series x 3 regimes = 45 tests of "the Sharpe ratio is the same in both states" (stationary
bootstrap of dates, mean block 10, 5,000 resamples); Benjamini-Hochberg over the 45.

Structural breaks
    Mean-shift sup-F test (15 % trimming, bootstrap p, 5,000 resamples) on each of the 15 gross series, BH over the
    15; the location of the maximum and the means either side are reported.
    Factor-space stability: on the names present in all four folds' universes with complete data, PCA is fitted every
    21 sessions on the trailing 504 (2013-01 to 2018-12); the overlap ||V1'V2||_F^2 / k of the leading-k subspaces
    (k = 5, 10) is averaged by lag in months, against the random-subspace level k / N and the overlap of
    consecutive fits; the first eigenvector (the market) is tracked by |cos|.

Hypotheses (priors written before running)
    H9e  Regimes: no Sharpe difference survives BH at 10 % across the 45 tests.  Prior: yes; with 45 tests about two
         nominal p < 0.05 are expected by chance even if nothing is there.  If anything, I expect PCA reversal to look
         better in high-volatility / high-dispersion states (liquidity provision is compensated in stress), but four
         years hold few stress days.
    H9f  Breaks: no series has a sup-F p < 0.05 after BH.  Prior: yes, and the test has little power (a fall of a
         Sharpe of 1 to 0 within four years is hard to see); a non-rejection is not evidence of stability.
    H9g  Factor space: the market direction is stable (|cos| > 0.95 at 12 months); the k = 5 space overlaps > 0.8 with
         itself one month apart and decays with lag; k = 10 is less stable than k = 5.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage9_regimes_breaks
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.stage8_multiple_testing import load_series
from statarb.backtest.pca_walkforward import prepare_pca_fold
from statarb.backtest.walkforward import make_folds
from statarb.config import load_config
from statarb.data.pipeline import DataPipeline
from statarb.models.pca import fit_pca
from statarb.research.registry import Registry
from statarb.selection.screening import ScreenConfig
from statarb.statistics.breaks import subspace_overlap, sup_mean_break_test
from statarb.statistics.multiple_testing import benjamini_hochberg
from statarb.statistics.regimes import causal_regimes, regime_sharpe_difference

R = Path("experiments/results")
SEED = 20260926
N_BOOT, MEAN_BLOCK = 5000, 10.0
FIT_WINDOW, STEP = 504, 21


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def factor_space_stability(pipe, tickers: list[str], universes: list[set[str]]) -> dict:
    common = sorted(set.intersection(*universes))
    panel = pipe.panel(common, "2011-01-03", "2018-12-31")
    ret = panel.ret.iloc[1:]  # the first row has no previous close
    ret = ret.loc[:, ret.notna().all()]  # names with a complete history over the whole window
    dates = list(range(FIT_WINDOW, len(ret), STEP))
    spaces = {5: [], 10: []}
    first_vec = []
    for t in dates:
        fit = fit_pca(ret.iloc[t - FIT_WINDOW : t].to_numpy())
        for k in spaces:
            spaces[k].append(fit.loadings(k))
        first_vec.append(fit.components[:, 0])
    n_assets = ret.shape[1]
    out = {
        "n_common_names": n_assets,
        "n_fits": len(dates),
        "first_fit": str(ret.index[dates[0]].date()),
        "last_fit": str(ret.index[dates[-1]].date()),
    }
    for k, vs in spaces.items():
        by_lag: dict[int, list[float]] = {}
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                by_lag.setdefault(j - i, []).append(subspace_overlap(vs[i], vs[j]))
        out[f"k{k}"] = {
            "random_level_k_over_N": k / n_assets,
            "overlap_by_lag_months": {
                str(lag): float(np.mean(v))
                for lag, v in sorted(by_lag.items())
                if lag in (1, 3, 6, 12, 24, 36, 48, 60)
            },
        }
    by_lag = {}
    for i in range(len(first_vec)):
        for j in range(i + 1, len(first_vec)):
            by_lag.setdefault(j - i, []).append(abs(float(first_vec[i] @ first_vec[j])))
    out["first_eigenvector_abs_cosine_by_lag_months"] = {
        str(lag): float(np.mean(v))
        for lag, v in sorted(by_lag.items())
        if lag in (1, 3, 6, 12, 24, 36, 48, 60)
    }
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    gross, net, pairs_cfg, pca_cfg = load_series()
    pipe = DataPipeline(load_config("configs/data_default.yaml"))
    folds = make_folds(pipe.cfg.split, pipe.calendar, 2015, 2018, 4, "rolling")
    assert all(f.phase == "research" for f in folds)
    fds = [prepare_pca_fold(pipe, f, ScreenConfig()) for f in folds]
    tickers0 = list(fds[0].tickers)
    panel = pipe.panel(tickers0, "2011-01-03", "2018-12-31")
    ew = panel.ret.mean(axis=1, skipna=True)
    disp = panel.ret.std(axis=1)
    reg = causal_regimes(ew, disp, 21, 126, 252).reindex(gross.index)
    assert reg.notna().all().all(), "regime labels are missing inside the research window"

    out = {
        "experiment": "stage9_regimes_breaks",
        "seed": SEED,
        "git_commit": git_commit(),
        "n_days": len(gross),
    }
    out["regime_share_of_days"] = {c: float(reg[c].mean()) for c in reg}
    # ---- regimes ---------------------------------------------------------------------------
    print("Regimes ...", flush=True)
    rows = []
    for c in gross.columns:
        for rname in reg.columns:
            t = regime_sharpe_difference(
                gross[c].to_numpy(), reg[rname].to_numpy(), N_BOOT, MEAN_BLOCK, SEED
            )
            rows.append(
                {
                    "strategy": c,
                    "regime": rname,
                    **t,
                    "net_sharpe_true": regime_sharpe_difference(
                        net[c].to_numpy(), reg[rname].to_numpy(), 200, MEAN_BLOCK, SEED
                    )["sharpe_true"],
                    "net_sharpe_false": regime_sharpe_difference(
                        net[c].to_numpy(), reg[rname].to_numpy(), 200, MEAN_BLOCK, SEED
                    )["sharpe_false"],
                }
            )
    df = pd.DataFrame(rows)
    rej, adj = benjamini_hochberg(df["p_value"].to_numpy(), 0.10)
    df["p_bh"] = adj
    out["regime_tests"] = df.to_dict(orient="records")
    out["regime_summary"] = {
        "n_tests": len(df),
        "n_p_below_5pct": int((df["p_value"] < 0.05).sum()),
        "expected_by_chance_at_5pct": 0.05 * len(df),
        "n_bh_rejections_at_10pct": int(rej.sum()),
        "smallest_p": float(df["p_value"].min()),
        "smallest_p_test": df.loc[df["p_value"].idxmin(), ["strategy", "regime"]].tolist(),
    }
    out["gross_sharpe_by_year"] = {
        c: {
            str(y): float(g.mean() / g.std(ddof=1) * np.sqrt(252))
            for y, g in gross[c].groupby(gross.index.year)
        }
        for c in gross.columns
    }
    print(
        "  regime tests: nominal p<0.05:",
        out["regime_summary"]["n_p_below_5pct"],
        "of",
        len(df),
        "| BH@10%:",
        int(rej.sum()),
        flush=True,
    )

    # ---- structural breaks -----------------------------------------------------------------
    print("Break tests ...", flush=True)
    brows = []
    for c in gross.columns:
        b = sup_mean_break_test(gross[c].to_numpy(), 0.15, N_BOOT, MEAN_BLOCK, SEED)
        brows.append(
            {
                "strategy": c,
                **b,
                "break_date": str(gross.index[b["index"]].date()),
                "sharpe_before": b["mean_before"]
                / gross[c].iloc[: b["index"]].std(ddof=1)
                * np.sqrt(252),
                "sharpe_after": b["mean_after"]
                / gross[c].iloc[b["index"] :].std(ddof=1)
                * np.sqrt(252),
            }
        )
    bdf = pd.DataFrame(brows)
    brej, badj = benjamini_hochberg(bdf["p_value"].to_numpy(), 0.05)
    bdf["p_bh"] = badj
    out["break_tests"] = bdf.to_dict(orient="records")
    out["break_summary"] = {
        "n_tests": len(bdf),
        "n_p_below_5pct": int((bdf["p_value"] < 0.05).sum()),
        "n_bh_rejections_at_5pct": int(brej.sum()),
        "smallest_p": float(bdf["p_value"].min()),
    }
    print(
        "  break tests: nominal p<0.05:",
        out["break_summary"]["n_p_below_5pct"],
        "| BH@5%:",
        int(brej.sum()),
        flush=True,
    )

    print("Factor-space stability ...", flush=True)
    out["factor_space"] = factor_space_stability(pipe, tickers0, [set(fd.tickers) for fd in fds])
    fingerprint = pipe.fingerprint()
    pipe.close()
    out["data_fingerprint"] = fingerprint
    (R / "stage9_regimes_breaks.json").write_text(json.dumps(out, indent=2, default=float))
    Registry().register(
        stage=9,
        kind="diagnostic",
        name="regimes_breaks_factor_stability_research_phase",
        phases=["research"],
        windows={"test_years": [2015, 2018]},
        strategy="none (regime, break and factor-space diagnostics on existing series)",
        parameters={
            "n_boot": N_BOOT,
            "mean_block": MEAN_BLOCK,
            "regimes": ["vol_high", "trend_up", "disp_high"],
        },
        seed=SEED,
        data_fingerprint=fingerprint,
        git_commit=git_commit(),
        notes="adds no trial; research phase only",
    )
    print("wrote", R / "stage9_regimes_breaks.json")


if __name__ == "__main__":
    sys.exit(main())
