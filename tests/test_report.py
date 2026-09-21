"""The report machinery: templating semantics, the registry table, and consistency with saved results."""

from __future__ import annotations

from pathlib import Path

import pytest

from statarb.research.registry import Registry
from statarb.research.report import (
    headline_numbers,
    load_results,
    money,
    registry_markdown,
    render,
)

RESULTS = Path("experiments/results")
needs_results = pytest.mark.skipif(
    not (RESULTS / "stage9_capacity.json").exists(), reason="no saved results"
)


def test_render_fills_placeholders_with_default_and_explicit_formats():
    nums = {"x": 1.23456, "n": 7, "s": "abc", "p": 0.0375, "flag": True}
    out = render("{{x}} | {{ x|.3f }} | {{n}} | {{s}} | {{p|.1%}} | {{n|+d}} | {{flag}}", nums)
    assert out == "1.23 | 1.235 | 7 | abc | 3.8% | +7 | True"
    assert render("no placeholders {}", nums) == "no placeholders {}"


def test_render_refuses_a_number_it_does_not_have():
    with pytest.raises(KeyError, match="unknown number"):
        render("value {{missing}}", {"x": 1})


def test_money_formatting():
    assert money(0) == "none" and money(float("inf")) == "unbounded"
    assert money(2.3e6) == "$2.3M" and money(247e3) == "$247K" and money(1.5e9) == "$1.5B"


def test_registry_markdown_keeps_every_record_including_void_runs(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    reg.register(
        stage=7,
        kind="diagnostic",
        name="control",
        phases=["research"],
        results={"s": 1.23456},
        notes="run 1 | void",
    )
    reg.register(
        stage=7,
        kind="diagnostic",
        name="control",
        phases=["research"],
        results={"s": 2.0},
        notes="corrected",
    )
    md = registry_markdown(reg.records())
    lines = md.strip().splitlines()
    assert len(lines) == 2 + 2  # header, rule, two records
    assert (
        "`control`" in lines[2] and "s=1.23" in lines[2] and "run 1 / void" in lines[2]
    )  # pipes are escaped
    assert lines[3].split("|")[5].strip() == "2"  # the second run is marked run 2


@needs_results
def test_headline_numbers_are_read_from_the_results_and_are_self_consistent():
    reg = Registry()
    counts = {"strategy_distinct": reg.count_trials("strategy")["distinct_specs"]}
    n = headline_numbers(load_results(), counts)
    assert n["registry.strategy_distinct"] == 24 and n["d.best"] == "pca_reversal_k5"
    assert n["d.gross_min"] <= n["d.best_gross"] == n["d.gross_max"]
    assert n["d.best_ci_lo"] < n["d.best_gross"] < n["d.best_ci_hi"]
    assert n["e.qualifiers"] == 0 and n["f.survives"] == "no" and n["h.pca_positive_capacity"] == 0
    assert n["a.calibrated_discoveries"] < n["a.expected_null"]
    assert n["f.neff_lo"] < n["f.neff_hi"] < 24 and n["f.dsr24"] < n["f.dsr_neff_lo"] < 0.95
    assert n["c.surface_dsr"] < n["f.dsr24"]  # more trials can only lower the deflated Sharpe
    assert n["h.analytic_check"] < 1e-9


@needs_results
def test_the_committed_readme_and_research_log_are_a_fresh_rendering_of_the_saved_results():
    template, readme = Path("docs/README.template.md"), Path("README.md")
    if not template.exists():
        pytest.skip("report not built yet")
    reg = Registry()
    counts = {
        "strategy_distinct": reg.count_trials("strategy")["distinct_specs"],
        "strategy_runs": reg.count_trials("strategy")["total_runs"],
    }
    numbers = headline_numbers(load_results(), counts)
    assert readme.read_text() == render(template.read_text(), numbers)
    assert Path("docs/research_log.md").read_text().endswith(registry_markdown(reg.records()))
