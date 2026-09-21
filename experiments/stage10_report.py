"""Stage 10: render README.md (from docs/README.template.md) and docs/research_log.md from saved results.

Nothing is computed here: every number in the report comes from ``experiments/results`` through
``statarb.research.report.headline_numbers``; the research log is the registry, rendered.  A test fails if the
committed files differ from a fresh rendering.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage10_report
"""

from __future__ import annotations

import sys
from pathlib import Path

from statarb.research.registry import Registry
from statarb.research.report import headline_numbers, load_results, registry_markdown, render

LOG_HEADER = """# Research log

The append-only experiment registry (`experiments/registry.jsonl`), rendered. Every trial evaluated on data is here,
including diagnostics, simulations, re-runs and **voided** runs (a re-run of the same specification gets `Run` 2, and the
first run stays: nothing is edited or deleted). `Kind` decides how a record is counted: only `strategy` records evaluated
on real data enter the multiple-testing count (Stage 8); `diagnostic` and `simulation` records are logged but not counted.
Phases: `research` = 2011-2018, `validation` = 2019-2021, `holdout` = 2022 onward. The only record that lists validation or holdout is the
Stage 2 diagnostic `engle_granger_null_size_real_data` (size of a test on independent shifted pairs, nothing selected; it predates the
holdout guard); no strategy, screen or parameter has used them.

"""


def main() -> None:
    reg = Registry()
    counts = {
        "strategy_distinct": reg.count_trials("strategy")["distinct_specs"],
        "strategy_runs": reg.count_trials("strategy")["total_runs"],
    }
    numbers = headline_numbers(load_results(), counts)
    Path("README.md").write_text(render(Path("docs/README.template.md").read_text(), numbers))
    Path("docs/research_log.md").write_text(LOG_HEADER + registry_markdown(reg.records()))
    used = {r["name"] for r in reg.records() if {"validation", "holdout"} & set(r["phases"])}
    allowed = {"engle_granger_null_size_real_data"}  # the disclosed Stage 2 size diagnostic
    assert used <= allowed, (
        f"a registry record other than the disclosed exception used validation/holdout: {used - allowed}"
    )
    assert all(r["kind"] == "diagnostic" for r in reg.records() if r["name"] in allowed)
    print("wrote README.md and docs/research_log.md;", len(reg.records()), "registry records")


if __name__ == "__main__":
    sys.exit(main())
