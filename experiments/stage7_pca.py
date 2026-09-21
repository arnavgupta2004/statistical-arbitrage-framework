"""Stage 7 experiment: PCA statistical arbitrage on the RESEARCH phase only (test years 2015-2018).

Fixed before the grid was run.  Validation (2019-2021) and the holdout (2022+) are not touched.

Disclosure.  While building the pipeline, the 2015 fold was run once on two configurations to check
that the book was sane (neutrality, turnover, sizing).  That run reported turnover ~150-300x capital a
year and costs several times the gross P&L; nothing in the specification below was changed on its
basis: every constant is a literature value (Avellaneda-Lee thresholds, 60-day windows, kappa filter)
or a size fixed on first principles (2 % of capital per name at median residual vol), and the grid
was written down before the first full run.

Correction.  The first full run of this script used a control that was NOT dollar-neutral: the
projection returned the weights unchanged when there were no factors (net exposure reached 6.5x
capital), contradicting the specification below.  The projection was fixed (and a test that asserted
the wrong behaviour replaced), and the script was re-run in full.  Only the control is affected (the
grid has k >= 5 and was neutral in both runs, with identical results).  The registry keeps both runs
of the control (``run`` 1 and 2 of the same spec); the first is void.

Protocol
    Folds      the same four calendar-year blocks as Stages 5-6 (test 2015 / 2016 / 2017 / 2018,
               rolling 4-year training window), same point-in-time, identity-verified, liquid, gap-free
               universe as the pair screen (``eligible_tickers`` at ``train_end``), fixed for the block.
    Model      PCA of standardised daily returns, refitted every 21 sessions on the trailing 504
               (never on the day it scores); rolling 60-day no-intercept betas on the factors;
               residual of the scored day is out-of-sample.  Positions flat at each block's end.
    Signals    two definitions (``sscore``: Avellaneda-Lee OU s-score; ``reversal``: 5-day residual
               reversal), both "positive = rich".  Enter |score| > 1.25, exit inside 0.5, stop at 4,
               time stop 60 days.  Sized 2 % of capital per name (x median sigma_E / sigma_E, cap 2.5),
               exactly dollar- and factor-neutral, gross floats with the number of open names.
    Grid (6)   signal in {sscore, reversal} x n_factors in {5, 10, 15}.   All 6 are registered trials.
    Control    ``reversal`` with n_factors = 0 (raw return; only dollar neutral): does removing factors
               add anything?  A diagnostic: it cannot be selected.
    Costs      the Stage 6 central model on the same causal ADV / volatility inputs, capital $100M;
               weights are fractions of total capital, so returns are on capital and comparable to Stage 6.
    Placebo    200 draws per fold: the score columns are relabelled by a random permutation of the names
               that are tradable on most test days, identical for every day of the block, then sizing,
               neutralisation and costs run on the stock that holds the position.  Same construction,
               same turnover, zero expected gross.  Draws are shared across configurations (paired).
    Rule       pre-specified finalist rule, identical to Stage 6: net Sharpe > 0 AND net Sharpe >= the
               median of the config's own net placebo; the best qualifier (by net Sharpe) advances.  If
               none qualifies, nothing advances and validation is not run.

Hypotheses (priors from the literature and Stages 3-6 written down before the run)
    H7a  Residual reversal is real gross: the ``reversal`` configs' gross Sharpe is above the 95th
         percentile of their placebo.  Prior: yes (Khandani-Lo, Avellaneda-Lee), much weaker than the
         published Sharpes of 1-1.5 (the effect is known to have decayed).
    H7b  Net of the Stage 6 costs at $100M every configuration has negative Sharpe (turnover of
         100-300x capital a year against a few bps of gross per trade).  Prior: strong.
    H7c  The factor model adds something: gross Sharpe of reversal with k = 5, 10, 15 exceeds the
         k = 0 control.  Prior: weak (market and sector moves are removed by the neutral projection
         even for k = 0 only partly).
    H7d  (capacity, descriptive and cannot change the rule)  net Sharpe at capital $10M vs $100M:
         impact is superlinear in size, so a smaller book keeps more of its gross.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage7_pca
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
from statarb.backtest.costs import CostConfig
from statarb.backtest.pca_walkforward import (
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
from statarb.signals.residuals import ResidualConfig
from statarb.statistics.bootstrap import bootstrap_ci

SEED = 20260923
FIRST_YEAR, LAST_YEAR, TRAIN_YEARS = 2015, 2018, 4
N_DRAWS = 200
CAPITAL = 1e8
GRID = [
    PCAStrategyConfig(residual=ResidualConfig(n_factors=k, signal=s))
    for s in ("sscore", "reversal")
    for k in (5, 10, 15)
]
CONTROL = PCAStrategyConfig(residual=ResidualConfig(n_factors=0, signal="reversal"))
CAPITALS = (1e7, 1e8, 1e9)
COST_SCALES = (0.0, 0.25, 0.5, 1.0, 2.0)
OUT = Path("experiments/results")


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def stitch(fss, cfg, cost, perms=None) -> pd.DataFrame:
    parts = [
        run_fold(fs, cfg, cost, None if perms is None else perms[fs.fd.fold.index]).assign(
            fold=fs.fd.fold.index
        )
        for fs in fss
    ]
    return pd.concat(parts)


def evaluate(label, cfg, fss_by_cfg, cost, perms_by_draw, registry, fingerprint, kind):
    fss = fss_by_cfg[label]
    d = stitch(fss, cfg, cost)
    gross, net = d["pnl"].to_numpy(), d["net"].to_numpy()
    sh_g, ci_g, _ = bootstrap_ci(gross, metrics.sharpe, n_boot=2000, mean_block=10, seed=1)
    sh_n, ci_n, _ = bootstrap_ci(net, metrics.sharpe, n_boot=2000, mean_block=10, seed=1)
    pg, pn = [], []
    for perms in perms_by_draw:
        p = stitch(fss, cfg, cost, perms)
        pg.append(metrics.sharpe(p["pnl"].to_numpy()))
        pn.append(metrics.sharpe(p["net"].to_numpy()))
    pg, pn = np.array(pg), np.array(pn)
    comps = d[["cost_spread", "cost_commission", "cost_impact", "cost_borrow"]].sum()
    years = len(d) / 252.0
    res = {
        "config": cfg.model_dump(),
        "gross": metrics.performance_summary(gross, d["trade"].to_numpy()),
        "net": metrics.performance_summary(net, d["trade"].to_numpy()),
        "gross_sharpe_ci95": list(ci_g),
        "net_sharpe_ci95": list(ci_n),
        "sharpe_by_fold": {
            int(f): {
                "gross": float(metrics.sharpe(g["pnl"].to_numpy())),
                "net": float(metrics.sharpe(g["net"].to_numpy())),
            }
            for f, g in d.groupby("fold")
        },
        "book": {
            "avg_gross_exposure": float(d["gross_exposure"].mean()),
            "avg_names_held_incl_hedge": float(d["n_names"].mean()),
            "avg_signal_names": float(d["n_signal"].mean()),
            "turnover_per_year": float(d["trade"].sum() / years),
            "max_abs_net_exposure": float(d["net_exposure"].abs().max()),
            "share_days_participation_over_10pct": float((d["participation"] > 0.10).mean()),
        },
        "cost_bps_per_year": {k: float(v * 1e4 / years) for k, v in comps.items()},
        "gross_mean_bps_per_year": float(d["pnl"].sum() * 1e4 / years),
        "net_bps_per_year_if_zero_impact": float(
            (
                d["pnl"].sum()
                - comps["cost_spread"]
                - comps["cost_commission"]
                - comps["cost_borrow"]
            )
            * 1e4
            / years
        ),
        "breakeven_cost_multiple": float(d["pnl"].sum() / d["cost"].sum())
        if d["cost"].sum()
        else None,
        "placebo": {
            "n_draws": len(pg),
            "gross_sharpe_median": float(np.median(pg)),
            "gross_sharpe_5_95": [float(np.quantile(pg, 0.05)), float(np.quantile(pg, 0.95))],
            "gross_percentile": float((pg < sh_g).mean()),
            "net_sharpe_median": float(np.median(pn)),
            "net_sharpe_5_95": [float(np.quantile(pn, 0.05)), float(np.quantile(pn, 0.95))],
            "net_percentile": float((pn < sh_n).mean()),
        },
    }
    sens = {}
    for cap in CAPITALS:
        dd = stitch(fss, cfg, CostConfig(capital=cap, slots=20))
        sens[f"capital_{cap:.0e}"] = float(metrics.sharpe(dd["net"].to_numpy()))
    res["net_sharpe_by_capital"] = sens
    res["net_sharpe_by_cost_scale"] = {
        str(s): float(metrics.sharpe((d["pnl"] - s * d["cost"]).to_numpy())) for s in COST_SCALES
    }
    registry.register(
        stage=7,
        kind=kind,
        name=f"walkforward_{label}",
        phases=["research"],
        windows={
            "test_years": [FIRST_YEAR, LAST_YEAR],
            "train_years": TRAIN_YEARS,
            "mode": "rolling",
        },
        universe="S&P 500 point-in-time, identity-verified, pair-screen eligibility",
        strategy="PCA residual mean reversion, dollar/factor neutral",
        parameters={"strategy": cfg.model_dump()},
        features=["log dividend-adjusted returns", "PCA factors refit every 21d on 504d"],
        costs={"model": "stage6 central", "capital": CAPITAL},
        results={
            "gross_sharpe": sh_g,
            "net_sharpe": sh_n,
            "gross_placebo_percentile": res["placebo"]["gross_percentile"],
            "net_placebo_percentile": res["placebo"]["net_percentile"],
        },
        seed=SEED,
        data_fingerprint=fingerprint,
        git_commit=git_commit(),
        notes="research phase only; validation and holdout untouched"
        + (
            "; no-factor control, diagnostic. Run 1 (before the k=0 dollar-neutrality fix) is void; "
            "this is the corrected run."
            if kind == "diagnostic"
            else "; unaffected by the control fix, re-run for a consistent artefact"
        ),
    )
    return res, d, sh_g, sh_n


def main() -> None:
    warnings.filterwarnings("ignore")
    pipe = DataPipeline(load_config("configs/data_default.yaml"))
    split = pipe.cfg.split
    folds = make_folds(split, pipe.calendar, FIRST_YEAR, LAST_YEAR, TRAIN_YEARS, "rolling")
    assert all(f.phase == "research" for f in folds), "Stage 7 evaluates the research phase only"
    cost = CostConfig(capital=CAPITAL, slots=20)
    fds = []
    for f in folds:
        fd = prepare_pca_fold(pipe, f, ScreenConfig())
        fds.append(fd)
        print(
            f"fold {f.index}: test {f.test_start.date()}..{f.test_end.date()}  universe {len(fd.tickers)}",
            flush=True,
        )

    configs = {c.label: c for c in GRID + [CONTROL]}
    fss_by_cfg = {label: [fold_signals(fd, c, cost) for fd in fds] for label, c in configs.items()}
    fit_info = {
        label: [
            {
                "fold": fs.fd.fold.index,
                "universe": len(fs.fd.tickers),
                "explained_variance_mean": float(np.mean([x["explained"] for x in fs.sig.fits]))
                if fs.sig.fits
                else None,
                "marchenko_pastur_k_mean": float(np.mean([x["mp_k"] for x in fs.sig.fits]))
                if fs.sig.fits
                else None,
                "score_nan_share": float(np.isnan(fs.sig.score).mean()),
            }
            for fs in fss
        ]
        for label, fss in fss_by_cfg.items()
    }
    # the same permutations for every configuration (paired placebo)
    ref = fss_by_cfg[GRID[0].label]
    perms_by_draw = [
        {
            fs.fd.fold.index: placebo_permutation(
                fs, np.random.default_rng([SEED, fs.fd.fold.index, d])
            )
            for fs in ref
        }
        for d in range(N_DRAWS)
    ]

    registry, fingerprint = Registry(), pipe.fingerprint()
    results, daily = {}, []
    for label, cfg in configs.items():
        kind = "diagnostic" if cfg is CONTROL else "strategy"
        res, d, sh_g, sh_n = evaluate(
            label, cfg, fss_by_cfg, cost, perms_by_draw, registry, fingerprint, kind
        )
        res["kind"] = kind
        results[label] = res
        daily.append(d.assign(config=label))
        p = res["placebo"]
        print(
            f"{label:20s} gross {sh_g:+.2f} (placebo pct {p['gross_percentile']:.0%}, med {p['gross_sharpe_median']:+.2f}) | "
            f"net {sh_n:+.2f} (placebo med {p['net_sharpe_median']:+.2f}, pct {p['net_percentile']:.0%}) | "
            f"turnover/yr {res['book']['turnover_per_year']:.0f}x | breakeven cost x{res['breakeven_cost_multiple']:.2f}",
            flush=True,
        )

    # finalist rule, exactly as pre-specified (grid only; the control cannot be selected)
    qualifiers = [
        label
        for label in (c.label for c in GRID)
        if results[label]["net"]["sharpe"] > 0
        and results[label]["net"]["sharpe"] >= results[label]["placebo"]["net_sharpe_median"]
    ]
    best = max(qualifiers, key=lambda x: results[x]["net"]["sharpe"]) if qualifiers else None
    print(
        "finalist rule: qualifiers =",
        qualifiers,
        "-> advances:",
        best or "NOTHING (validation not run)",
    )

    # descriptive only: relation to the pair strategy and to the equal-weight market
    extra = {}
    stage5 = OUT / "stage5_daily_returns.csv"
    if stage5.exists():
        p5 = pd.read_csv(stage5, parse_dates=["date"])
        pairs_gross = p5.groupby("date")["net"].mean()  # Stage 5 series are gross
        for label in (c.label for c in GRID):
            d = pd.concat(daily)[lambda x, lab=label: x["config"] == lab]
            j = d["pnl"].to_frame("pca").join(pairs_gross.rename("pairs"), how="inner")
            extra[label] = {"corr_gross_with_mean_pairs_gross": float(j["pca"].corr(j["pairs"]))}
    ew = pd.concat([fd.ret.iloc[fd.first :].mean(axis=1).rename("ew") for fd in fds])
    for label in (c.label for c in GRID):
        d = pd.concat(daily)[lambda x, lab=label: x["config"] == lab]
        j = d["pnl"].to_frame("pnl").join(ew, how="inner")
        b = float(np.polyfit(j["ew"], j["pnl"], 1)[0])
        extra.setdefault(label, {})["market_beta_vs_equal_weight"] = b
        extra[label]["corr_with_equal_weight"] = float(j["pnl"].corr(j["ew"]))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(daily).to_csv(OUT / "stage7_daily_returns.csv", index_label="date")
    out = {
        "experiment": "stage7_pca",
        "seed": SEED,
        "git_commit": git_commit(),
        "data_fingerprint": fingerprint,
        "split": split.model_dump(mode="json"),
        "capital": CAPITAL,
        "n_draws": N_DRAWS,
        "grid": [c.label for c in GRID],
        "control": CONTROL.label,
        "folds": [
            {
                "fold": f.index,
                "test": [str(f.test_start.date()), str(f.test_end.date())],
                "train": [str(f.train_start.date()), str(f.train_end.date())],
            }
            for f in folds
        ],
        "fit_info": fit_info,
        "results": results,
        "descriptive": extra,
        "finalist": {
            "rule": "net Sharpe > 0 and >= median of its own net placebo",
            "qualifiers": qualifiers,
            "advances": best,
        },
        "phases_evaluated": ["research"],
        "note": "validation and holdout untouched",
    }
    (OUT / "stage7_pca.json").write_text(json.dumps(out, indent=2, default=str))
    print("wrote", OUT / "stage7_pca.json")
    pipe.close()


if __name__ == "__main__":
    sys.exit(main())
