"""Stage 9 experiment A: parameter sensitivity and stability of the pair relationships (RESEARCH phase).

Fixed before it was run.  Validation (2019-2021) and the holdout (2022+) are not touched.

What this is, and is not.  Every variant below is an evaluation of a configuration on the 2015-2018
research data, i.e. a *look*.  This is a sensitivity analysis: the whole surface is reported, nothing is
chosen from it, and **no variant can advance**: the finalist rule of Stages 6-7 found no qualifier and a
sensitivity surface cannot create one (a variant that looked good would be a hypothesis for a future,
registered trial, on data not yet used).  The surface is registered as a diagnostic; the number of
configurations in it is reported, together with what the deflated Sharpe of its best point would be if that
point were treated as one more trial.

Anchors (fixed a priori)
    PCA    ``pca_reversal_k5`` (the Stage 7 result under test) and ``pca_sscore_k10`` (the class defaults).
    Pairs  ``StrategyConfig()`` defaults: static hedge, entry 2.0, exit 0.5, stop 4.0, max hold 60, z-window 60.

One-at-a-time variations (each dimension moved alone; the anchor's own value is the centre)
    PCA    n_factors {3, 5, 8, 10, 15, 20}; fit window {378, 504, 756} (*); refit every {5, 21, 63}; beta window
           {40, 60, 90}; reversal horizon {3, 5, 10} (reversal) / kappa filter {4.2, 8.4, 16.8} (s-score);
           entry {1.0, 1.25, 1.5, 2.0}; exit {0.25, 0.5, 0.75}; stop {3, 4, 6}; max hold {20, 60};
           name size {1 %, 2 %, 4 %}.  Plus the two-way grid entry x n_factors for the reversal anchor.
    Pairs  hedge {static, expanding, Kalman 1e-5, rolling 250}; entry {1, 1.5, 2, 2.5, 3}; exit {0, 0.25, 0.5,
           1}; stop {3, 4, 6}; max hold {20, 60, 120}; z-window {30, 60, 120, 250}; and the *screen's*
           thresholds, re-selected from the saved screen tables without re-screening: calibrated alpha {0.01,
           0.05, 0.10, 0.20}, half-life band {(5, 60), (2, 120), (10, 40)}, portfolio size {10, 20, 40}
           (slots = size).
    (*) The first draft of this list had 252 for the fit window; the run stopped at once because 252 days is fewer than
    the 308-361 names (a singular correlation matrix, which ``fit_pca`` refuses).  No result had been produced;
    the value was replaced by 378, the smallest that exceeds the largest universe.
    Everything is walk-forward on the same four folds (test years 2015-2018), the Stage 6 central cost
    model at $100M, gross and net Sharpe reported.

Stability of the pair relationships (hypothesis B), all four folds: for each selected pair and for the
baseline pool (calibrated p > 0.05, beta > 0), the training hedge ratio and mean are frozen and the spread is
examined over the test year with the Stage 3 metrics (ADF p, sd ratio, mean shift, AR(1), half-life); plus the
rank correlation between a pair's training and out-of-sample cointegration strength.

Decision rule for "robust" (PCA anchor): at least 70 % of the one-at-a-time neighbours have positive gross
Sharpe **and** their median is at least half the anchor's.  Otherwise "fragile".

Hypotheses (priors written before running)
    H9a  The PCA reversal gross effect is not an isolated peak: it passes the "robust" rule.  Prior: yes for the
         parameters that do not remove the residual (windows, thresholds); the number of factors is a gradient
         (Stage 7: Sharpe falls as k grows).
    H9b  No variant of either family has a positive *net* Sharpe at $100M (turnover is 100-300x).  Prior: yes.
    H9c  The pair strategy has no region of the parameter space with a stable positive gross Sharpe.  Prior: yes.
    H9d  Pair relationships are not stable out of sample: the selected pairs' out-of-sample ADF rejection rate is
         not above the baseline pool's, and in-sample cointegration strength has no rank correlation with
         out-of-sample strength.  Prior: yes (Stage 3: 0/20 vs 10.5 %).

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage9_sensitivity
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiments.stage3_pair_discovery import group_summary, oos_metrics, wilson
from statarb.backtest import metrics
from statarb.backtest.costs import CostConfig, PairCostModel, build_market_data
from statarb.backtest.pca_walkforward import (
    PCAStrategyConfig,
    fold_signals,
    prepare_pca_fold,
    run_fold,
)
from statarb.backtest.walkforward import (
    StrategyConfig,
    aggregate,
    make_folds,
    prepare_fold,
    run_pairs,
    walk_forward,
)
from statarb.config import load_config
from statarb.data.pipeline import DataPipeline
from statarb.research.registry import Registry
from statarb.selection.screening import ScreenConfig, reselect_pairs
from statarb.signals.residuals import ResidualConfig
from statarb.statistics.sharpe import deflated_sharpe, sharpe_moments

R = Path("experiments/results")
SEED = 20260925
STAGE5_SEED = 20260922
FIRST_YEAR, LAST_YEAR, TRAIN_YEARS = 2015, 2018, 4
COST = CostConfig(capital=1e8, slots=20)
N_TRIALS_REGISTERED = 24
ROBUST_SHARE, ROBUST_MEDIAN = 0.70, 0.5

RES_KEYS = {
    "n_factors",
    "fit_window",
    "refit_every",
    "beta_window",
    "signal",
    "reversal_days",
    "kappa_min",
}
BOOK_KEYS = {"entry", "exit", "stop", "max_hold", "name_size"}


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


# ---- PCA ---------------------------------------------------------------------------------------
def pca_variant(base: PCAStrategyConfig, **kw) -> PCAStrategyConfig:
    unknown = set(kw) - RES_KEYS - BOOK_KEYS
    if unknown:
        raise ValueError(unknown)
    return PCAStrategyConfig(
        residual=base.residual.model_copy(update={k: v for k, v in kw.items() if k in RES_KEYS}),
        book=base.book.model_copy(update={k: v for k, v in kw.items() if k in BOOK_KEYS}),
    )


def pca_dimensions(anchor: PCAStrategyConfig) -> dict[str, list]:
    dims = {  # book-only dimensions first: they reuse the anchor's cached signals
        "entry": [1.0, 1.25, 1.5, 2.0],
        "exit": [0.25, 0.5, 0.75],
        "stop": [3.0, 4.0, 6.0],
        "max_hold": [20, 60],
        "name_size": [0.01, 0.02, 0.04],
        "n_factors": [3, 5, 8, 10, 15, 20],
        "fit_window": [378, 504, 756],
        "refit_every": [5, 21, 63],
        "beta_window": [40, 60, 90],
    }
    if anchor.residual.signal == "reversal":
        dims["reversal_days"] = [3, 5, 10]
    else:
        dims["kappa_min"] = [4.2, 8.4, 16.8]
    return dims


class PCARunner:
    """Runs PCA variants on the four folds; signals are cached for the last two residual configs
    only (they hold betas of shape days x names x k), so book-only variants reuse them."""

    def __init__(self, fds):
        self.fds, self.cache, self.evaluated = fds, {}, set()

    def signals(self, cfg):
        key = cfg.residual.model_dump_json()
        if key not in self.cache:
            while len(self.cache) >= 2:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = [fold_signals(fd, cfg, COST) for fd in self.fds]
        return self.cache[key]

    def __call__(self, cfg: PCAStrategyConfig) -> dict:
        self.evaluated.add(cfg.model_dump_json())
        d = pd.concat([run_fold(fs, cfg, COST) for fs in self.signals(cfg)])
        years = len(d) / 252.0
        return {
            "gross_sharpe": float(metrics.sharpe(d["pnl"].to_numpy())),
            "net_sharpe": float(metrics.sharpe(d["net"].to_numpy())),
            "turnover_per_year": float(d["trade"].sum() / years),
            "gross_bps_per_year": float(d["pnl"].sum() * 1e4 / years),
            "cost_bps_per_year": float(d["cost"].sum() * 1e4 / years),
            "daily_gross": d["pnl"].to_numpy(),
        }


def surface_summary(rows: list[dict], anchor_key: str) -> dict:
    neigh = [r for r in rows if r["dimension"] != "anchor" and not r["is_anchor_value"]]
    g = np.array([r["gross_sharpe"] for r in neigh])
    n = np.array([r["net_sharpe"] for r in neigh])
    anchor = next(r for r in rows if r["is_anchor_value"])
    return {
        "anchor": anchor_key,
        "anchor_gross_sharpe": anchor["gross_sharpe"],
        "anchor_net_sharpe": anchor["net_sharpe"],
        "n_neighbours": len(neigh),
        "share_gross_positive": float((g > 0).mean()),
        "share_gross_at_least_half_anchor": float((g >= 0.5 * anchor["gross_sharpe"]).mean()),
        "median_gross": float(np.median(g)),
        "min_gross": float(g.min()),
        "max_gross": float(g.max()),
        "rank_of_anchor_among_all": int(1 + (g > anchor["gross_sharpe"]).sum()),
        "share_net_positive": float((n > 0).mean()),
        "best_net": float(n.max()),
        "best_net_variant": neigh[int(np.argmax(n))]["variant"],
    }


def run_pca_surface(runner: PCARunner, anchor: PCAStrategyConfig, label: str) -> tuple[list, list]:
    rows, dailies = [], []
    seen: dict[str, dict] = {}

    def evaluate(cfg, dimension, value, is_anchor_value):
        key = cfg.model_dump_json()
        if key not in seen:
            seen[key] = runner(cfg)
        r = {k: v for k, v in seen[key].items() if k != "daily_gross"}
        rows.append(
            {
                "anchor": label,
                "dimension": dimension,
                "value": value,
                "variant": f"{dimension}={value}",
                "is_anchor_value": is_anchor_value,
                **r,
            }
        )
        dailies.append(seen[key]["daily_gross"])

    evaluate(anchor, "anchor", "-", True)
    for dim, values in pca_dimensions(anchor).items():
        current = (anchor.residual.model_dump() | anchor.book.model_dump())[dim]
        for v in values:
            if v == current:
                continue
            evaluate(pca_variant(anchor, **{dim: v}), dim, v, False)
    return rows, dailies


def two_way_grid(runner: PCARunner, anchor: PCAStrategyConfig) -> list[dict]:
    out = []
    for k, entry in product((3, 5, 8, 10, 15), (1.0, 1.25, 1.5, 2.0)):
        r = runner(pca_variant(anchor, entry=entry, n_factors=k))
        out.append(
            {"entry": entry, "n_factors": k, **{a: b for a, b in r.items() if a != "daily_gross"}}
        )
    return out


# ---- pairs -------------------------------------------------------------------------------------
def factory(cfg: CostConfig):
    def make(fd):
        return PairCostModel(
            build_market_data(fd.dollar_volume, fd.ret, fd.high, fd.low, fd.close, cfg), cfg
        )

    return make


def run_pair_variant(
    fds, strat: StrategyConfig, masks=None, cost: CostConfig | None = None
) -> dict:
    cost = cost or CostConfig(capital=1e8, slots=strat.slots)
    parts, n_trades = [], 0
    for fd in fds:
        pairs = fd.screen.pairs
        mask = pairs["selected"] if masks is None else masks(pairs)
        chosen = pairs[mask]
        index = fd.logp.loc[fd.fold.test_start : fd.fold.test_end].index
        runs = run_pairs(
            fd, strat, list(zip(chosen["y"], chosen["x"], strict=True)), factory(cost)(fd)
        )
        parts.append(aggregate(runs, index, strat.slots))
        n_trades += sum(len(r.trades) for r in runs)
    d = pd.concat(parts)
    years = len(d) / 252.0
    return {
        "gross_sharpe": float(metrics.sharpe(d["pnl"].to_numpy())),
        "net_sharpe": float(metrics.sharpe(d["net"].to_numpy())),
        "turnover_per_year": float(d["trade"].sum() / years),
        "gross_bps_per_year": float(d["pnl"].sum() * 1e4 / years),
        "cost_bps_per_year": float(d["cost"].sum() * 1e4 / years),
        "n_trades": n_trades,
        "daily_gross": d["pnl"].to_numpy(),
    }


def pair_variants() -> list[tuple[str, object, StrategyConfig, dict]]:
    base = StrategyConfig()
    out = []
    for m, kw in (
        ("static", {}),
        ("expanding", {}),
        ("kalman", {"kalman_delta": 1e-5}),
        ("rolling", {"hedge_window": 250}),
    ):
        out.append(("hedge", m, base.model_copy(update={"hedge_method": m, **kw}), {}))
    for dim, values in (
        ("entry", (1.0, 1.5, 2.0, 2.5, 3.0)),
        ("exit", (0.0, 0.25, 0.5, 1.0)),
        ("stop", (3.0, 4.0, 6.0)),
        ("max_hold", (20, 60, 120)),
        ("z_window", (30, 60, 120, 250)),
    ):
        for v in values:
            out.append((dim, v, base.model_copy(update={dim: v}), {}))
    for a in (0.01, 0.05, 0.10, 0.20):
        out.append(("screen_alpha", a, base, {"alpha": a}))
    for lo, hi in ((5, 60), (2, 120), (10, 40)):
        out.append(("screen_half_life", f"{lo}-{hi}", base, {"hl": (lo, hi)}))
    for k in (10, 20, 40):
        out.append(("portfolio_size", k, base.model_copy(update={"slots": k}), {"max_pairs": k}))
    return out


def pair_mask_fn(sel: dict):
    def fn(pairs):
        return reselect_pairs(
            pairs,
            sel.get("alpha", 0.05),
            *sel.get("hl", (5.0, 60.0)),
            sel.get("max_pairs", 20),
        )

    return fn


# ---- stability of pair relationships -----------------------------------------------------------
def relationship_stability(fds) -> dict:
    per_fold, all_rows = {}, []
    for fd in fds:
        pairs = fd.screen.pairs
        n_train = int((fd.logp.index <= fd.fold.train_end).sum())
        groups = {
            "selected": pairs[pairs["selected"]],
            "baseline_pool": pairs[(pairs["p_calibrated"] > 0.05) & (pairs["beta"] > 0)],
        }
        fold_out = {}
        for name, g in groups.items():
            rows = []
            for r in g.itertuples():
                s = fd.logp[r.y] - r.beta * fd.logp[r.x]
                if s.isna().any():
                    continue
                m = oos_metrics(s, n_train, r.spread_sd)
                rows.append(
                    {
                        "y": r.y,
                        "x": r.x,
                        "fold": fd.fold.index,
                        "group": name,
                        "p_calibrated": r.p_calibrated,
                        "T_train": r.T,
                        **m,
                    }
                )
            df = pd.DataFrame(rows)
            fold_out[name] = group_summary(df)
            all_rows.append(df)
        per_fold[str(fd.fold.test_start.year)] = fold_out
    allp = pd.concat(all_rows, ignore_index=True)
    pooled = {}
    for name, g in allp.groupby("group"):
        rej = int((g["oos_adf_p"] < 0.05).sum())
        pooled[name] = {
            "n_pairs": len(g),
            "share_oos_adf_rejects_5pct": rej / len(g),
            "wilson95": wilson(rej, len(g)),
            "median_oos_sd_ratio": float(g["oos_sd_ratio"].median()),
            "median_oos_mean_shift_sd": float(g["oos_mean_shift_sd"].median()),
            "median_oos_half_life": float(g["oos_half_life"].median()),
            "share_no_reversion_ar1_ge_1": float((g["oos_ar1_b"] >= 1.0).mean()),
        }
    ok = allp.dropna(subset=["oos_adf_p", "p_calibrated"])
    rho = stats.spearmanr(ok["p_calibrated"], ok["oos_adf_p"])
    pooled["rank_correlation_train_p_vs_oos_adf_p"] = {
        "spearman": float(rho.statistic),
        "p_value": float(rho.pvalue),
        "n": len(ok),
    }
    return {"per_fold": per_fold, "pooled": pooled}


def main() -> None:
    warnings.filterwarnings("ignore")
    pipe = DataPipeline(load_config("configs/data_default.yaml"))
    split = pipe.cfg.split
    folds = make_folds(split, pipe.calendar, FIRST_YEAR, LAST_YEAR, TRAIN_YEARS, "rolling")
    assert all(f.phase == "research" for f in folds), "Stage 9 evaluates the research phase only"

    print("PCA surfaces ...", flush=True)
    runner = PCARunner([prepare_pca_fold(pipe, f, ScreenConfig()) for f in folds])
    anchors = {
        "pca_reversal_k5": PCAStrategyConfig(
            residual=ResidualConfig(n_factors=5, signal="reversal")
        ),
        "pca_sscore_k10": PCAStrategyConfig(residual=ResidualConfig(n_factors=10, signal="sscore")),
    }
    out = {
        "experiment": "stage9_sensitivity",
        "seed": SEED,
        "git_commit": git_commit(),
        "pca": {},
        "pairs": {},
    }
    all_daily, n_configs = [], 0
    for label, anchor in anchors.items():
        rows, dailies = run_pca_surface(runner, anchor, label)
        summ = surface_summary(rows, label)
        summ["robust_rule_passed"] = bool(
            summ["share_gross_positive"] >= ROBUST_SHARE
            and summ["median_gross"] >= ROBUST_MEDIAN * summ["anchor_gross_sharpe"]
        )
        out["pca"][label] = {"summary": summ, "variants": rows}
        all_daily += dailies
        print(
            f"  {label}: anchor gross {summ['anchor_gross_sharpe']:+.2f}; {summ['n_neighbours']} neighbours, "
            f"{100 * summ['share_gross_positive']:.0f}% positive gross, median {summ['median_gross']:+.2f}, "
            f"best net {summ['best_net']:+.2f}; robust: {summ['robust_rule_passed']}",
            flush=True,
        )
    out["pca"]["two_way_entry_by_k_reversal"] = two_way_grid(runner, anchors["pca_reversal_k5"])
    n_configs += len(runner.evaluated)

    print("Pair screens and surface (screens re-run: several minutes) ...", flush=True)
    fds = [prepare_fold(pipe, f, ScreenConfig(seed=STAGE5_SEED, n_null_panels=10)) for f in folds]
    # the re-selection helper must reproduce the screen's own selection at the default thresholds
    for fd in fds:
        assert reselect_pairs(fd.screen.pairs, 0.05, 5.0, 60.0, 20).equals(
            fd.screen.pairs["selected"]
        )
    central = run_pair_variant(fds, StrategyConfig())
    check = walk_forward(fds, StrategyConfig(), cost_factory=factory(COST)).daily
    assert np.allclose(check["pnl"].to_numpy(), central["daily_gross"], atol=1e-12), (
        "engine mismatch"
    )
    rows = []
    for dim, value, strat, sel in pair_variants():
        r = run_pair_variant(fds, strat, pair_mask_fn(sel) if sel else None)
        rows.append(
            {
                "dimension": dim,
                "value": value,
                "variant": f"{dim}={value}",
                **{k: v for k, v in r.items() if k != "daily_gross"},
            }
        )
        all_daily.append(r["daily_gross"])
        print(
            f"  pairs {dim}={value}: gross {r['gross_sharpe']:+.2f} net {r['net_sharpe']:+.2f}",
            flush=True,
        )
    g = np.array([r["gross_sharpe"] for r in rows])
    out["pairs"]["central"] = {k: v for k, v in central.items() if k != "daily_gross"}
    out["pairs"]["variants"] = rows
    out["pairs"]["summary"] = {
        "n_variants": len(rows),
        "share_gross_positive": float((g > 0).mean()),
        "share_gross_above_0.5": float((g > 0.5).mean()),
        "max_gross": float(g.max()),
        "min_gross": float(g.min()),
        "median_gross": float(np.median(g)),
        "share_net_positive": float(np.mean([r["net_sharpe"] > 0 for r in rows])),
        "best_net": float(max(r["net_sharpe"] for r in rows)),
    }
    n_configs += len(rows)

    print("Stability of the pair relationships ...", flush=True)
    out["relationship_stability"] = relationship_stability(fds)

    # the multiple-testing weight of the surface, were its best point ever treated as a trial
    best = int(np.argmax([metrics.sharpe(x) for x in all_daily]))
    sr, sk, ku, n = sharpe_moments(all_daily[best])
    var = float(np.var([sharpe_moments(x)[0] for x in all_daily], ddof=1))
    out["surface_as_trials"] = {
        "n_configurations_evaluated": n_configs,
        "best_gross_sharpe_in_surface": sr * np.sqrt(252),
        "dsr_if_n_is_24_plus_surface": deflated_sharpe(
            sr, n, sk, ku, N_TRIALS_REGISTERED + n_configs, var
        )["dsr"],
        "sr0_annual_if_n_is_24_plus_surface": deflated_sharpe(
            sr, n, sk, ku, N_TRIALS_REGISTERED + n_configs, var
        )["sr0"]
        * np.sqrt(252),
    }

    fingerprint = pipe.fingerprint()
    pipe.close()
    out["data_fingerprint"] = fingerprint
    R.mkdir(parents=True, exist_ok=True)
    (R / "stage9_sensitivity.json").write_text(json.dumps(out, indent=2, default=float))
    Registry().register(
        stage=9,
        kind="diagnostic",
        name="parameter_sensitivity_surfaces_research_phase",
        phases=["research"],
        windows={"test_years": [FIRST_YEAR, LAST_YEAR], "train_years": TRAIN_YEARS},
        universe="S&P 500 point-in-time, identity-verified",
        strategy="sensitivity of the PCA and pair strategies (nothing selectable)",
        parameters={
            "n_configurations": n_configs,
            "anchors": list(anchors) + ["StrategyConfig defaults"],
        },
        results={
            "pca_reversal_robust": out["pca"]["pca_reversal_k5"]["summary"]["robust_rule_passed"]
        },
        seed=SEED,
        data_fingerprint=fingerprint,
        git_commit=git_commit(),
        notes="a sensitivity surface: every point is a look at research data, none can advance",
    )
    print("wrote", R / "stage9_sensitivity.json")


if __name__ == "__main__":
    sys.exit(main())
