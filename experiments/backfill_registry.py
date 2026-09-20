"""Back-fill the experiment registry with the trials run before it existed (Stages 2-4).

Each entry is derived from the experiment's own results file and marked as back-filled.  ``kind``
decides how a trial is counted later: real-data strategy variants count against a Sharpe claim in
Stage 8; simulations and methodological diagnostics are logged but not counted.  Idempotent: an
experiment already present (same name and spec) is skipped.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.backfill_registry
"""

from __future__ import annotations

import json
from pathlib import Path

from statarb.research.registry import Registry

R = Path("experiments/results")


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


def main() -> None:
    reg = Registry()
    have = {(r["name"], r["spec_id"]) for r in reg.records()}
    n_new = 0

    def add(**kw):
        nonlocal n_new
        from statarb.research.registry import spec_id

        probe = {
            k: kw.get(k)
            for k in (
                "stage",
                "kind",
                "name",
                "universe",
                "windows",
                "strategy",
                "parameters",
                "features",
                "costs",
                "seed",
            )
        }
        probe["costs"] = kw.get("costs") or {"model": "none (gross)"}
        probe["parameters"] = kw.get("parameters") or {}
        probe["windows"] = kw.get("windows") or {}
        probe["features"] = kw.get("features") or []
        if (kw["name"], spec_id(probe)) in have:
            return
        reg.register(**kw)
        n_new += 1

    s2 = load("stage2_null_size.json")
    add(
        stage=2,
        kind="diagnostic",
        name="engle_granger_null_size_real_data",
        phases=["research", "validation", "holdout"],
        windows={"windows": "2011-2015, 2016-2020, 2021-2025"},
        universe="survivors with full data",
        strategy="none (test calibration)",
        parameters={"horizons": [252, 1250], "trends": ["c", "ct"]},
        seed=s2["seed"],
        data_fingerprint=s2["data_fingerprint"],
        git_commit=s2["git_commit"],
        notes="back-filled. Predates the holdout guard: its 2021-2025 window overlaps the holdout; "
        "independent shifted pairs, no strategy or parameter selected from it.",
    )

    s3 = load("stage3_pair_discovery.json")
    add(
        stage=3,
        kind="strategy",
        name="pair_screen_default_2011_2015",
        phases=["research"],
        windows={"train": list(s3["train_window"]), "oos_look": s3["oos_window"]},
        universe="S&P 500 point-in-time, identity-verified",
        strategy="EG screen, empirical null",
        parameters=s3["screen_config"],
        features=["log dividend-adjusted price"],
        results={
            "n_candidates": s3["funnel"]["n_candidates"],
            "n_tests": s3["funnel"]["n_tests"],
            "calibrated_discoveries_5pct": s3["funnel"]["discoveries"]["0.05"],
        },
        seed=s3["seed"],
        data_fingerprint=s3["data_fingerprint"],
        git_commit=s3["git_commit"],
        notes="back-filled. One pre-specified window; 1,259 hypotheses (pairs) tested inside it.",
    )
    s3d = load("stage3_null_diagnostics.json")
    add(
        stage=3,
        kind="diagnostic",
        name="pair_screen_null_vs_real_posthoc",
        phases=["research"],
        windows={"train": list(s3["train_window"])},
        strategy="none (post-hoc explanatory)",
        parameters={"label": s3d["label"]},
        notes="back-filled; post-hoc; screen not changed on its basis",
    )

    s4s = load("stage4_hedge_ratio_synthetic.json")
    add(
        stage=4,
        kind="simulation",
        name="hedge_ratio_synthetic_prespecified",
        phases=[],
        strategy="pair strategy on simulated data",
        parameters={"methods": 8, "regimes": 3, "sims": s4s["sims"]},
        seed=s4s["seed"],
        git_commit=s4s["git_commit"],
        notes="back-filled; simulated, not counted as a trial",
    )
    s4p = load("stage4_hedge_ratio_synthetic_posthoc.json")
    add(
        stage=4,
        kind="simulation",
        name="hedge_ratio_synthetic_posthoc_mild_break",
        phases=[],
        strategy="pair strategy on simulated data",
        parameters={"methods": 8, "regimes": 1, "sims": s4p["sims"]},
        seed=s4p["seed"],
        git_commit=s4p["git_commit"],
        notes="back-filled; post-hoc regime; simulated",
    )

    s4r = load("stage4_real_pairs.json")
    for group in ("selected", "baseline_non_significant"):
        for label, res in s4r["results"][group].items():
            kind = (
                "strategy" if group == "selected" else "diagnostic"
            )  # 8 variants = 8 trials, not 16
            add(
                stage=4,
                kind=kind,
                name=f"real_pairs_2016_{group}_{label}",
                phases=["research"],
                windows={"train_end": s4r["train_end"], "oos": s4r["oos_window"]},
                universe="Stage 3 frozen pairs, 2016",
                strategy="z-score reversion, gross",
                parameters={"variant": label, "params": s4r["params"], "group": group},
                results={"gross_sharpe": res["sharpe"], "n_pairs": res["n_pairs"]},
                seed=s4r["screen_seed"],
                data_fingerprint=s4r["data_fingerprint"],
                git_commit=s4r["git_commit"],
                notes="back-filled. 2016 was research-phase data. The 8 variants are the trials; "
                "the "
                "baseline group re-runs them on a control set and is logged as a diagnostic.",
            )
    print(f"registered {n_new} new entries; strategy trials so far:", reg.count_trials("strategy"))


if __name__ == "__main__":
    main()
