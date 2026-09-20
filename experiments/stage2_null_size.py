"""Stage 2 experiment: is the Engle-Granger p-value calibrated on REAL equity prices?

Hypothesis
    Under a true null of "no cointegration", P(p < alpha) = alpha.  Synthetic Gaussian random
    walks satisfy this (tests/); real prices have fat tails and volatility clustering, which the
    theory ignores.

Design
    * NULL WITH REAL MARGINALS: pair stock i's actual log-price path with stock j's log *returns*,
      circularly shifted by a random offset and re-accumulated.  Each leg keeps its real marginal
      dynamics (fat tails, volatility clustering, autocorrelation) but the shared shocks (market
      days, sector news) are destroyed, so the series are independent by construction and the test
      SHOULD reject alpha of the time.
    * REAL PAIRS, same stocks, unshifted: what the test says about actual co-movement.
    * BEST OF TWO DIRECTIONS on the null: rejection rate of min(p_YX, p_XY).  This is the
      data-snooping inflation from testing both regressions, i.e. why engle_granger_both reports
      n_tests = 2.
    * Log prices are dividend-adjusted, anchored at the window end (causal), no imputation.
    * Both first-stage specifications are run: trend="c" and trend="ct".  HYPOTHESIS (stated after
      the first run showed 2-15 % null rejection at a nominal 5 %): real log prices drift, a
      drifting regressor detrends the residual, so the constant-only tables over-reject (Hansen
      1992) and "ct" should fix it.  RESULT: REFUTED -- "ct" does not remove the over-rejection.
    * Two controls (trend="c"): BOTH LEGS GAUSSIAN with each stock's mean and volatility (a harness
      check: must reject ~alpha), and REAL y vs GAUSSIAN x (does one real leg alone distort it?).

This is a size diagnostic, NOT a search for tradeable pairs: no pair is reported or ranked.

Data caveat
    Only tickers with full data in a window enter (i.e. survivors): fine for a null-calibration
    check, which does not need a point-in-time universe, but it must not be read as a statement
    about the S&P 500 as it was.  Pairs are NOT independent (each y carries the market factor, and
    stocks recur across pairings), so the Wilson intervals, which assume independence, are too
    narrow; the spread across windows is the more honest measure of uncertainty.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage2_null_size
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

from statarb.config import load_config
from statarb.data.pipeline import DataPipeline
from statarb.selection.cointegration import engle_granger

SEED = 20260920
WINDOWS = [("2011-01-03", "2015-12-31"), ("2016-01-04", "2020-12-31"), ("2021-01-04", "2025-12-31")]
HORIZONS = [252, 1250]  # trading days: one year, five years
PAIRINGS = 3  # independent random pairings of the eligible tickers, disjoint within a pairing
OUT = Path("experiments/results/stage2_null_size.json")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(centre - half), float(centre + half))


def rate(pvals: np.ndarray, alpha: float) -> dict:
    k, n = int((pvals < alpha).sum()), len(pvals)
    lo, hi = wilson(k, n)
    return {"alpha": alpha, "rejections": k, "n": n, "rate": k / n, "ci95": [lo, hi]}


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def main() -> None:
    warnings.filterwarnings("ignore")
    # This experiment PREDATES the holdout guard and its 2021-2025 window overlaps what is now the
    # sealed holdout.  It used independent shifted pairs (a size diagnostic): no strategy,
    # parameter, feature or universe was selected from it.  The guard is therefore switched off
    # here, and the overlap is disclosed in the README.  Its conclusion (miscalibration) is
    # visible in the 2011-2015 and 2016-2020 windows alone.
    cfg = load_config("configs/data_default.yaml").model_copy(update={"split": None})
    pipe = DataPipeline(cfg)
    manifest = pipe.store.manifest()
    tickers = list(manifest.loc[manifest["status"] == "ok", "ticker"])
    rng = np.random.default_rng(SEED)  # pairings and shifts: the main experiment
    rng_ctrl = np.random.default_rng(
        SEED + 1
    )  # controls only, so they cannot perturb the main draws
    results = []
    for start, end in WINDOWS:
        panel = pipe.panel(tickers, start, end)  # as_of = end: what was knowable at window end
        elig = panel.eligible(start, end, 1.0)
        zero_vol = (panel.volume[elig] == 0).mean()
        elig = [t for t in elig if zero_vol[t] < 0.01]  # drop stale stubs (symbol reuse)
        logp = np.log(panel.tr_close()[elig])
        print(f"window {start}..{end}: {len(elig)} eligible tickers", flush=True)
        for horizon in HORIZONS:
            lp = logp.iloc[-horizon:]
            ret = lp.diff().iloc[1:]
            n = len(lp)
            trends = ("c", "ct")
            acc = {t: {"null": [], "null_best": [], "real": [], "real_best": []} for t in trends}
            ctrl = {"both_gaussian": [], "real_y_gaussian_x": []}
            for _ in range(PAIRINGS):
                order = rng.permutation(len(elig))
                for a, b in zip(order[0::2], order[1::2], strict=False):
                    ya, xb = lp.iloc[:, a].to_numpy(), lp.iloc[:, b].to_numpy()
                    shift = int(rng.integers(n // 4, 3 * n // 4))
                    shifted = np.roll(ret.iloc[:, b].to_numpy(), shift)
                    x_null = np.concatenate([[xb[0]], xb[0] + np.cumsum(shifted)])
                    try:  # controls (trend="c"): Gaussian legs matched to mean / volatility
                        mu_a, sd_a = np.diff(ya).mean(), np.diff(ya).std(ddof=1)
                        mu_b, sd_b = ret.iloc[:, b].mean(), ret.iloc[:, b].std(ddof=1)
                        x_g = xb[0] + np.concatenate(
                            [[0.0], np.cumsum(rng_ctrl.normal(mu_b, sd_b, n - 1))]
                        )
                        y_g = ya[0] + np.concatenate(
                            [[0.0], np.cumsum(rng_ctrl.normal(mu_a, sd_a, n - 1))]
                        )
                        ctrl["both_gaussian"].append(engle_granger(y_g, x_g).pvalue)
                        ctrl["real_y_gaussian_x"].append(engle_granger(ya, x_g).pvalue)
                    except ValueError:
                        pass
                    for t in trends:
                        try:
                            fwd = engle_granger(ya, x_null, trend=t).pvalue
                            rev = engle_granger(x_null, ya, trend=t).pvalue
                            rf = engle_granger(ya, xb, trend=t).pvalue
                            rr = engle_granger(xb, ya, trend=t).pvalue
                        except ValueError:
                            continue
                        acc[t]["null"].append(fwd)
                        acc[t]["null_best"].append(min(fwd, rev))
                        acc[t]["real"].append(rf)
                        acc[t]["real_best"].append(min(rf, rr))
            controls = {k: [rate(np.array(v), a) for a in (0.05, 0.01)] for k, v in ctrl.items()}
            print(
                f"  n={horizon:5d} CONTROLS trend=c | both legs Gaussian @5% "
                f"{controls['both_gaussian'][0]['rate']:.3f} | real y vs Gaussian x @5% "
                f"{controls['real_y_gaussian_x'][0]['rate']:.3f}",
                flush=True,
            )
            for t in trends:
                d = {k: np.array(v) for k, v in acc[t].items()}
                row = {
                    "window": [start, end],
                    "horizon_days": horizon,
                    "trend": t,
                    "n_tickers": len(elig),
                    "n_pairs": len(d["null"]),
                    "null_real_marginals": [rate(d["null"], a) for a in (0.05, 0.01)],
                    "null_best_of_two_directions": [rate(d["null_best"], a) for a in (0.05, 0.01)],
                    "real_pairs": [rate(d["real"], a) for a in (0.05, 0.01)],
                    "real_pairs_best_of_two": [rate(d["real_best"], a) for a in (0.05, 0.01)],
                    "controls_trend_c": controls,
                }
                results.append(row)
                print(
                    f"  n={horizon:5d} trend={t:2s} pairs={row['n_pairs']:4d} | "
                    f"null@5% {row['null_real_marginals'][0]['rate']:.3f} "
                    f"null@1% {row['null_real_marginals'][1]['rate']:.3f} | "
                    f"best-of-2 null@5% {row['null_best_of_two_directions'][0]['rate']:.3f} | "
                    f"real pairs@5% {row['real_pairs'][0]['rate']:.3f}",
                    flush=True,
                )
    out = {
        "experiment": "stage2_null_size",
        "seed": SEED,
        "git_commit": git_commit(),
        "data_fingerprint": pipe.fingerprint(),
        "pairings": PAIRINGS,
        "results": results,
        "note": "size diagnostic on survivors; no pair is ranked or reported",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)
    pipe.close()


if __name__ == "__main__":
    sys.exit(main())
