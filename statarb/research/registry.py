"""Machine-readable experiment registry: an append-only JSON-lines log of every trial.

A *trial* is one configuration evaluated on data.  The number of trials is the input the
multiple-testing corrections (Stage 8: deflated Sharpe, reality check) need, and it is only honest
if failed and uninteresting trials are recorded too.  Nothing is ever edited or deleted: a re-run
appends a second record with the same ``spec_id`` (a hash of what was run, not of its results),
and ``count_trials`` reports both distinct configurations and total runs.

Record fields: id, date, stage, kind (``strategy`` | ``diagnostic`` | ``simulation``), name,
universe, phases used (research / validation / holdout), training / validation / test windows,
strategy, parameters, features, cost assumptions, results, seed, git commit, data fingerprint,
notes.

``kind`` matters for counting: only ``strategy`` trials evaluated on real data enter the
multiple-testing count for a Sharpe ratio; simulations and methodological diagnostics are logged
but not counted against a strategy claim.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

KINDS = ("strategy", "diagnostic", "simulation")
PHASES = ("research", "validation", "holdout")
SPEC_FIELDS = (
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


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def spec_id(record: dict) -> str:
    """Stable id of *what was run* (results, dates and commit excluded)."""
    spec = {k: record.get(k) for k in SPEC_FIELDS}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:12]


class Registry:
    def __init__(self, path: str | Path = "experiments/registry.jsonl"):
        self.path = Path(path)

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def register(
        self,
        *,
        stage: int,
        kind: str,
        name: str,
        phases: list[str],
        windows: dict | None = None,
        universe: str | None = None,
        strategy: str | None = None,
        parameters: dict | None = None,
        features: list[str] | None = None,
        costs: dict | None = None,
        results: dict | None = None,
        seed: int | None = None,
        data_fingerprint: str | None = None,
        git_commit: str | None = None,
        notes: str = "",
    ) -> dict:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        bad = [p for p in phases if p not in PHASES]
        if bad:
            raise ValueError(f"unknown phases {bad}; expected a subset of {PHASES}")
        record = {
            "date": pd.Timestamp.now(tz="UTC").isoformat(),
            "stage": stage,
            "kind": kind,
            "name": name,
            "universe": universe,
            "phases": sorted(phases),
            "windows": windows or {},
            "strategy": strategy,
            "parameters": parameters or {},
            "features": features or [],
            "costs": costs or {"model": "none (gross)"},
            "results": results or {},
            "seed": seed,
            "git_commit": git_commit or _git_commit(),
            "data_fingerprint": data_fingerprint,
            "notes": notes,
        }
        record["spec_id"] = spec_id(record)
        record["run"] = 1 + sum(r["spec_id"] == record["spec_id"] for r in self.records())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return record

    def count_trials(self, kind: str = "strategy", phase: str | None = None) -> dict:
        """Distinct configurations and total runs, optionally only trials that used ``phase``."""
        recs = [
            r
            for r in self.records()
            if r["kind"] == kind and (phase is None or phase in r["phases"])
        ]
        return {"distinct_specs": len({r["spec_id"] for r in recs}), "total_runs": len(recs)}
