"""Assemble the final report from saved results, so its numbers cannot drift from them.

``headline_numbers`` reads the result files of Stages 3-9 and returns a flat dictionary of named
numbers; ``render`` fills a template's ``{{name}}`` / ``{{name|format}}`` placeholders from it (a
missing name is an error, never a silent blank); ``registry_markdown`` renders the append-only
experiment registry as the research log.  ``experiments/stage10_report.py`` writes ``README.md`` and
``docs/research_log.md`` from them, and a test fails if the committed files differ from a fresh
rendering.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

FILES = {
    "s3": "stage3_pair_discovery.json",
    "s5": "stage5_walkforward.json",
    "s6": "stage6_costs.json",
    "s7": "stage7_pca.json",
    "s7c": "stage7_placebo_checks.json",
    "s8": "stage8_multiple_testing.json",
    "s8b": "stage8_rc_scale_check.json",
    "s9a": "stage9_sensitivity.json",
    "s9b": "stage9_regimes_breaks.json",
    "s9c": "stage9_capacity.json",
    "s9d": "stage9_regime_followup.json",
}
PCA_GRID = [
    "pca_sscore_k5",
    "pca_sscore_k10",
    "pca_sscore_k15",
    "pca_reversal_k5",
    "pca_reversal_k10",
    "pca_reversal_k15",
]
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.]+)\s*(?:\|\s*([^}]*?))?\s*\}\}")


def load_results(directory: str | Path = "experiments/results") -> dict:
    d = Path(directory)
    return {k: json.loads((d / f).read_text()) for k, f in FILES.items()}


def money(x: float) -> str:
    if x == 0:
        return "none"
    if x == float("inf"):
        return "unbounded"
    if x >= 1e9:
        return f"${x / 1e9:.1f}B"
    return f"${x / 1e6:.1f}M" if x >= 1e6 else f"${x / 1e3:.0f}K"


def _rng(values) -> tuple[float, float]:
    v = list(values)
    return float(min(v)), float(max(v))


def headline_numbers(res: dict, registry_counts: dict | None = None) -> dict:
    """Named numbers used by the report; every one is read from a saved result file."""
    n: dict = {}
    s3, s5, s6, s7, s7c = res["s3"], res["s5"], res["s6"], res["s7"], res["s7c"]
    s8, s8b, a, b, c, d = res["s8"], res["s8b"], res["s9a"], res["s9b"], res["s9c"], res["s9d"]
    # ---- data and search space
    n["fingerprint"] = s7["data_fingerprint"]
    n["days"] = s8["n_days"]
    n["years"] = s8["selection_luck_table"]["sample_years"]
    if registry_counts:
        n.update({f"registry.{k}": v for k, v in registry_counts.items()})
    # ---- A: pair discovery
    f = s3["funnel"]
    n["a.candidates"], n["a.eligible"] = f["n_candidates"], f["n_eligible"]
    n["a.naive_discoveries"] = f["naive_discoveries_at_5pct"]
    n["a.calibrated_discoveries"], n["a.expected_null"] = (
        f["discoveries"]["0.05"],
        f["expected_false"]["0.05"],
    )
    n["a.oos_sel_reject"] = s3["oos_by_group"]["selected"]["share_oos_adf_rejects_5pct"]
    n["a.oos_sel_n"] = s3["oos_by_group"]["selected"]["n"]
    n["a.oos_base_reject"] = s3["oos_by_group"]["not_significant_baseline"][
        "share_oos_adf_rejects_5pct"
    ]
    n["a.oos_sel_halflife"] = s3["oos_by_group"]["selected"]["median_oos_half_life"]
    n["a.oos_sel_sd_ratio"] = s3["oos_by_group"]["selected"]["median_oos_sd_ratio"]
    n["a.pairs_configs"] = len(s5["results"])
    n["a.wf_gross_min"], n["a.wf_gross_max"] = _rng(
        v["summary"]["sharpe"] for v in s5["results"].values()
    )
    n["a.wf_pct_min"], n["a.wf_pct_max"] = _rng(
        v["placebo"]["screened_percentile"] for v in s5["results"].values()
    )
    n["a.wf_gross_best_ci_lo"] = min(v["sharpe_ci95"][0] for v in s5["results"].values())
    n["a.wf_gross_best_ci_hi"] = max(v["sharpe_ci95"][1] for v in s5["results"].values())
    n["a.pairs_rc_p"] = s8["family_gross"]["pairs_9"]["reality_check_p"]
    n["a.pairs_spa_p"] = s8["family_gross"]["pairs_9"]["spa_p"]["consistent"]
    n["a.bh_screens_max"] = max(v["bh_q0.05"] for v in s8["screen_fdr"].values())
    n["a.pi0_min"] = min(v["storey_pi0"]["0.5"] for v in s8["screen_fdr"].values())
    # ---- B: stability
    rel = a["relationship_stability"]["pooled"]
    n["b.sel_reject"], n["b.pool_reject"] = (
        rel["selected"]["share_oos_adf_rejects_5pct"],
        rel["baseline_pool"]["share_oos_adf_rejects_5pct"],
    )
    n["b.sel_pairs"], n["b.pool_pairs"] = (
        rel["selected"]["n_pairs"],
        rel["baseline_pool"]["n_pairs"],
    )
    n["b.sel_sd_ratio"], n["b.pool_sd_ratio"] = (
        rel["selected"]["median_oos_sd_ratio"],
        rel["baseline_pool"]["median_oos_sd_ratio"],
    )
    n["b.spearman"] = rel["rank_correlation_train_p_vs_oos_adf_p"]["spearman"]
    n["b.spearman_p"] = rel["rank_correlation_train_p_vs_oos_adf_p"]["p_value"]
    fs = b["factor_space"]
    for lag in ("1", "12", "60"):
        n[f"b.k5_overlap_{lag}m"] = fs["k5"]["overlap_by_lag_months"][lag]
    n["b.market_cos_12m"] = fs["first_eigenvector_abs_cosine_by_lag_months"]["12"]
    n["b.breaks_p_below_5"], n["b.breaks_n"] = (
        b["break_summary"]["n_p_below_5pct"],
        b["break_summary"]["n_tests"],
    )
    n["b.breaks_min_p"] = b["break_summary"]["smallest_p"]
    # ---- C: sensitivity
    for key, tag in (("pca_reversal_k5", "rev"), ("pca_sscore_k10", "ssc")):
        sm = a["pca"][key]["summary"]
        n[f"c.{tag}_anchor"], n[f"c.{tag}_neigh"] = sm["anchor_gross_sharpe"], sm["n_neighbours"]
        n[f"c.{tag}_share_pos"], n[f"c.{tag}_median"] = (
            sm["share_gross_positive"],
            sm["median_gross"],
        )
        n[f"c.{tag}_min"], n[f"c.{tag}_max"], n[f"c.{tag}_best_net"] = (
            sm["min_gross"],
            sm["max_gross"],
            sm["best_net"],
        )
    ps = a["pairs"]["summary"]
    n["c.pairs_variants"], n["c.pairs_share_pos"] = ps["n_variants"], ps["share_gross_positive"]
    n["c.pairs_best_gross"], n["c.pairs_best_net"], n["c.pairs_median"] = (
        ps["max_gross"],
        ps["best_net"],
        ps["median_gross"],
    )
    n["c.surface_configs"], n["c.surface_dsr"] = (
        a["surface_as_trials"]["n_configurations_evaluated"],
        a["surface_as_trials"]["dsr_if_n_is_24_plus_surface"],
    )
    rev = {
        v["variant"]: v
        for v in a["pca"]["pca_reversal_k5"]["variants"]
        if v["dimension"] != "anchor"
    }
    n["c.rev_k3"], n["c.rev_k20"] = (
        rev["n_factors=3"]["gross_sharpe"],
        rev["n_factors=20"]["gross_sharpe"],
    )
    # ---- D: PCA
    r7 = s7["results"]
    g = {k: r7[k]["gross"]["sharpe"] for k in PCA_GRID}
    n["d.gross_min"], n["d.gross_max"] = _rng(g.values())
    best = max(g, key=g.get)
    n["d.best"] = best
    n["d.best_gross"] = g[best]
    n["d.best_ci_lo"], n["d.best_ci_hi"] = r7[best]["gross_sharpe_ci95"]
    n["d.best_placebo_pct"] = r7[best]["placebo"]["gross_percentile"]
    n["d.control_gross"] = r7["pca_reversal_k0"]["gross"]["sharpe"]
    n["d.control_pct"] = r7["pca_reversal_k0"]["placebo"]["gross_percentile"]
    n["d.n_over_95"] = sum(
        r7[k]["placebo"]["gross_percentile"] >= 0.95
        for k in PCA_GRID
        if k.startswith("pca_reversal")
    )
    n["d.turnover_min"], n["d.turnover_max"] = _rng(
        r7[k]["book"]["turnover_per_year"] for k in PCA_GRID
    )
    n["d.universe_min"], n["d.universe_max"] = _rng(x["universe"] for x in s7["fit_info"][best])
    n["d.placebo_sd_ratio"] = (
        1.0 / s7c["B_real_data_symmetry_pca_reversal_k5"]["std_daily_pnl"]["placebo_over_real"]
    )
    n["d.mp_k_min"], n["d.mp_k_max"] = _rng(
        x["marchenko_pastur_k_mean"] for x in s7["fit_info"][best]
    )
    # ---- E: costs
    n["e.pairs_net_min"], n["e.pairs_net_max"] = _rng(
        v["net"]["net_sharpe"] for v in s6["results"].values()
    )
    n["e.pca_net_min"], n["e.pca_net_max"] = _rng(r7[k]["net"]["sharpe"] for k in PCA_GRID)
    n["e.pairs_turnover_min"], n["e.pairs_turnover_max"] = _rng(
        v["net"]["turnover_per_year"] for v in s6["results"].values()
    )
    n["e.pca_cost_min"], n["e.pca_cost_max"] = _rng(
        sum(r7[k]["cost_bps_per_year"].values()) for k in PCA_GRID
    )
    n["e.pca_gross_bps_min"], n["e.pca_gross_bps_max"] = _rng(
        r7[k]["gross_mean_bps_per_year"] for k in PCA_GRID
    )
    n["e.pca_breakeven_min"], n["e.pca_breakeven_max"] = _rng(
        r7[k]["breakeven_cost_multiple"] for k in PCA_GRID
    )
    n["e.pairs_breakeven_max"] = max(
        v["cost"]["breakeven_multiple"] for v in s6["results"].values()
    )
    n["e.impact_share"] = np.mean(
        [v["cost"]["components_share"]["cost_impact"] for v in s6["results"].values()]
    )
    n["e.qualifiers"] = len(s6["qualifying"]) + len(s7["finalist"]["qualifiers"])
    # ---- F: multiple testing
    bd = s8["gross"][best]
    n["f.psr"], n["f.dsr24"], n["f.sr0_24"] = bd["psr_vs_zero"], bd["dsr_n24"], bd["sr0_annual_n24"]
    n["f.dsr_neff_lo"], n["f.dsr_neff_hi"] = bd["dsr_neff_part"], bd["dsr_neff_corr"]
    n["f.neff_lo"], n["f.neff_hi"] = (
        s8["effective_trials_gross"]["n_eff_participation_ratio"],
        s8["effective_trials_gross"]["n_eff_average_correlation"],
    )
    n["f.rc_all"], n["f.spa_all"] = (
        s8["family_gross"]["all_15"]["reality_check_p"],
        s8["family_gross"]["all_15"]["spa_p"]["consistent"],
    )
    n["f.rc_pca"], n["f.spa_pca"] = (
        s8["family_gross"]["pca_6"]["reality_check_p"],
        s8["family_gross"]["pca_6"]["spa_p"]["consistent"],
    )
    n["f.rc_pca_equalvol"] = s8b["families"]["pca_6"]["reality_check_p_equal_volatility"]
    n["f.rw_min"] = min(
        v["p_romano_wolf"] for v in s8["family_gross"]["all_15"]["per_strategy"].values()
    )
    n["f.pbo_pca"], n["f.pbo_all"] = (
        s8["pbo_gross"]["pca_6"]["blocks16"]["pbo"],
        s8["pbo_gross"]["all_15"]["blocks16"]["pbo"],
    )
    n["f.survives"] = "yes" if s8["decision"]["survives"] else "no"
    n["f.min_backtest_years_24"] = s8["selection_luck_table"][
        "min_backtest_years_for_annual_sharpe_1"
    ]["24"]
    n["f.mintrl_years"] = bd["min_track_record_years_vs_zero"]
    # ---- G: regimes
    rs = b["regime_summary"]
    n["g.tests"], n["g.nominal"], n["g.expected"], n["g.bh"] = (
        rs["n_tests"],
        rs["n_p_below_5pct"],
        rs["expected_by_chance_at_5pct"],
        rs["n_bh_rejections_at_10pct"],
    )
    disp = {(t["strategy"], t["regime"]): t for t in b["regime_tests"]}
    n["g.k5_high"], n["g.k5_low"] = (
        disp[(best, "disp_high")]["sharpe_true"],
        disp[(best, "disp_high")]["sharpe_false"],
    )
    n["g.k5_p"] = disp[(best, "disp_high")]["p_value"]
    n["g.k5_net_high"] = disp[(best, "disp_high")]["net_sharpe_true"]
    n["g.shift_p"] = d["shift_test"][best]["p_value"]
    n["g.episodes"] = d["episodes_disp_high"]["n_true_episodes"]
    n["g.years_same_sign"] = sum(
        v["sharpe_high"] > v["sharpe_low"] for v in d["by_year_pca"][best].values()
    )
    # ---- H: capacity
    cs = c["strategies"]
    n["h.pca_fixed_min"], n["h.pca_fixed_max"] = _rng(
        cs[k]["fixed_cost_bps_per_year"] for k in PCA_GRID
    )
    n["h.pca_gross_min"], n["h.pca_gross_max"] = _rng(cs[k]["gross_bps_per_year"] for k in PCA_GRID)
    n["h.pca_positive_capacity"] = sum(cs[k]["break_even_capital"] > 0 for k in PCA_GRID)
    n["h.best_fixed_multiple"] = cs[best]["fixed_cost_multiple_for_break_even_at_vanishing_size"]
    n["h.best_capital_half_fixed"] = money(
        cs[best]["break_even_capital_by_fixed_cost_multiple"]["0.5"]
    )
    pos = {k: v["break_even_capital"] for k, v in cs.items() if v["break_even_capital"] > 0}
    n["h.pairs_positive"] = len(pos)
    n["h.pairs_capacity_max"], n["h.pairs_capacity_min"] = (
        money(max(pos.values())),
        money(min(pos.values())),
    )
    n["h.breach_capital_min"] = money(
        min(cs[k]["capital_at_which_5pct_of_days_breach_10pct_adv"] for k in PCA_GRID)
    )
    n["h.breach_capital_max"] = money(
        max(cs[k]["capital_at_which_5pct_of_days_breach_10pct_adv"] for k in PCA_GRID)
    )
    n["h.analytic_check"] = c["max_abs_diff_vs_stage7_reruns"]
    return n


def render(template: str, numbers: dict) -> str:
    """Fill ``{{name}}`` / ``{{name|spec}}`` placeholders; ``spec`` is a Python format spec."""

    def sub(m: re.Match) -> str:
        key, spec = m.group(1), m.group(2)
        if key not in numbers:
            raise KeyError(f"template uses an unknown number {key!r}")
        v = numbers[key]
        if spec:
            return format(v, spec)
        if isinstance(v, bool | str):
            return str(v)
        return f"{v:d}" if isinstance(v, int | np.integer) else f"{v:.2f}"

    return _PLACEHOLDER.sub(sub, template)


def registry_markdown(records: list[dict]) -> str:
    """The research log: one row per registry record (nothing is dropped, voided runs included)."""
    head = "| # | Stage | Kind | Name | Run | Phases | Seed | Commit | Key results | Notes |"
    rows = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(records, 1):
        res = ", ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in list(r.get("results", {}).items())[:4]
        )
        note = (r.get("notes") or "").replace("|", "/").replace("\n", " ")
        cells = [
            str(i),
            str(r["stage"]),
            r["kind"],
            f"`{r['name']}`",
            str(r.get("run", 1)),
            ",".join(r["phases"]) or "-",
            str(r.get("seed") or "-"),
            r.get("git_commit") or "-",
            res or "-",
            note[:160],
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"
