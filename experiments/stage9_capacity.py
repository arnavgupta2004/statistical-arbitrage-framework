"""Stage 9 experiment C: capacity, from the saved cost components (no re-run of any strategy).

Fixed before it was run.  Research phase only; nothing is evaluated on new data; a diagnostic, adds no trial.

Method: in the cost model, per unit of capital the spread, commission and borrow costs do not depend on the
capital, market impact scales with sqrt(capital) and participation with capital (``backtest/capacity.py``).
The saved daily components at $100M therefore give the net return at any capital and multiplier exactly.  The
analytic result is asserted to reproduce Stage 7's own net Sharpe at $10M and $1B (which were separate re-runs).

Questions, for the 15 strategies of Stages 5-7 and the Stage 7 control
    1  Break-even capital: the capital at which the mean net return is zero (0 if the strategy loses even with no
       impact).
    2  What would fixed costs (spread + commission + borrow) have to be?  The multiple of those costs at which the
       strategy breaks even at vanishing size, and the break-even capital for fixed-cost multiples 0.75, 0.5, 0.25, 0.
    3  How large can it get before more than 5 % of days breach 10 % of a name's ADV (PCA strategies, which saved
       participation)?

Hypothesis (prior written before running)
    H9h  No strategy breaks even at any capital under the central spread, commission and borrow assumptions
         (break-even capital 0 for all 15).  Prior: yes for the PCA family (zero-impact net was -124 to -1044 bps a
         year); for the pairs (cost 1.6-4.1 %/yr vs gross <= 1.4 %/yr) also yes.  Even with fixed costs halved the
         best PCA book is expected to need a capital below $10M.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage9_capacity
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from statarb.backtest import metrics
from statarb.backtest.capacity import (
    COMPONENTS,
    break_even_capital,
    net_at_capital,
    participation_breach_share,
)
from statarb.research.registry import Registry

R = Path("experiments/results")
BASE = 1e8
CAPITALS = (1e6, 1e7, 1e8, 1e9, 1e10)
FIXED_MULTS = (1.0, 0.75, 0.5, 0.25, 0.0)


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def main() -> None:
    s6 = pd.read_csv(R / "stage6_daily_net_returns.csv", parse_dates=["date"])
    s7 = pd.read_csv(R / "stage7_daily_returns.csv", parse_dates=["date"])
    s7j = json.loads((R / "stage7_pca.json").read_text())
    frames = {c: g.set_index("date") for c, g in s6.groupby("config")}
    frames |= {c: g.set_index("date") for c, g in s7.groupby("config")}

    # the analytic scaling must reproduce Stage 7's independent re-runs at other capitals
    worst = 0.0
    for c in s7j["grid"] + [s7j["control"]]:
        for cap, key in ((1e7, "capital_1e+07"), (1e9, "capital_1e+09")):
            got = metrics.sharpe(net_at_capital(frames[c], cap, BASE).to_numpy())
            worst = max(worst, abs(got - s7j["results"][c]["net_sharpe_by_capital"][key]))
    assert worst < 1e-9, f"analytic capacity disagrees with Stage 7's re-runs by {worst}"

    out = {
        "experiment": "stage9_capacity",
        "git_commit": git_commit(),
        "base_capital": BASE,
        "max_abs_diff_vs_stage7_reruns": worst,
        "strategies": {},
    }
    for c, d in frames.items():
        years = len(d) / 252.0
        comp = {k: float(d[k].sum() * 1e4 / years) for k in COMPONENTS}
        gross = float(d["pnl"].sum() * 1e4 / years)
        fixed = comp["cost_spread"] + comp["cost_commission"] + comp["cost_borrow"]
        row = {
            "gross_bps_per_year": gross,
            "fixed_cost_bps_per_year": fixed,
            "impact_bps_per_year_at_100M": comp["cost_impact"],
            "net_bps_per_year_at_zero_impact": gross - fixed,
            "fixed_cost_multiple_for_break_even_at_vanishing_size": gross / fixed
            if gross > 0
            else 0.0,
            "break_even_capital": break_even_capital(d, BASE),
            "break_even_capital_by_fixed_cost_multiple": {
                str(m): break_even_capital(d, BASE, spread_mult=m, commission_mult=m, borrow_mult=m)
                for m in FIXED_MULTS
            },
            "net_sharpe_by_capital_x_fixed_cost_multiple": {
                f"{cap:.0e}": {
                    str(m): float(
                        metrics.sharpe(
                            net_at_capital(
                                d, cap, BASE, spread_mult=m, commission_mult=m, borrow_mult=m
                            ).to_numpy()
                        )
                    )
                    for m in FIXED_MULTS
                }
                for cap in CAPITALS
            },
        }
        if "participation" in d:
            row["share_days_over_10pct_adv_by_capital"] = {
                f"{cap:.0e}": participation_breach_share(d["participation"], cap, BASE)
                for cap in CAPITALS
            }
            # capital at which 5 % of days breach: the 95th percentile of the daily peak participation
            q = float(np.quantile(d["participation"], 0.95))
            row["capital_at_which_5pct_of_days_breach_10pct_adv"] = (
                BASE * 0.10 / q if q > 0 else float("inf")
            )
        out["strategies"][c] = row
    out["summary"] = {
        "n_strategies": len(out["strategies"]),
        "n_with_positive_break_even_capital": int(
            sum(v["break_even_capital"] > 0 for v in out["strategies"].values())
        ),
        "best_fixed_cost_multiple_for_break_even": max(
            v["fixed_cost_multiple_for_break_even_at_vanishing_size"]
            for v in out["strategies"].values()
        ),
    }
    (R / "stage9_capacity.json").write_text(json.dumps(out, indent=2, default=float))
    Registry().register(
        stage=9,
        kind="diagnostic",
        name="capacity_from_cost_components_research_phase",
        phases=["research"],
        strategy="none (analytic capacity of the 16 saved strategy series)",
        parameters={
            "base_capital": BASE,
            "capitals": list(CAPITALS),
            "fixed_cost_multiples": list(FIXED_MULTS),
        },
        results=out["summary"],
        git_commit=git_commit(),
        notes="analytic from saved components; verified against Stage 7 re-runs; adds no trial",
    )
    print(json.dumps(out["summary"], indent=1), "max diff vs Stage 7 re-runs:", worst)
    for c in s7j["grid"]:
        v = out["strategies"][c]
        print(
            f"{c:18s} gross {v['gross_bps_per_year']:6.0f}  fixed {v['fixed_cost_bps_per_year']:6.0f}  impact {v['impact_bps_per_year_at_100M']:6.0f}  "
            f"break-even x{v['fixed_cost_multiple_for_break_even_at_vanishing_size']:.2f}  BE capital(fixed x0.5) {v['break_even_capital_by_fixed_cost_multiple']['0.5']:.2e}"
        )


if __name__ == "__main__":
    sys.exit(main())
