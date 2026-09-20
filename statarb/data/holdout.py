"""Holdout enforcement: a sealed final test period and an append-only ledger of who opened it.

The final holdout must not influence strategy, parameter, feature, universe or model selection.
Good intentions do not enforce that; code does.  When a split is configured, every
research-facing read of the data (``DataPipeline.panel``, ``raw_close``, ``identity_report``)
refuses any request that reaches the holdout -- through its ``end`` *or* through ``as_of``, which
can smuggle later splits into a price basis -- unless the holdout has been explicitly unlocked,
and unlocking is recorded.

The ledger (JSON lines, append-only, committed with the repository) makes the use of the holdout
auditable: ``unlock`` is allowed **once**.  A second unlock raises ``HoldoutAlreadySpent`` unless
the caller passes ``allow_reopen=True``, which is recorded as a ``reopen`` event: from then on
the holdout is contaminated and any result on it must say so.

Not covered (by design): data *operations* (refresh, audit, snapshot) read the whole store because
data integrity is not a modelling choice; and code that bypasses the pipeline and reads Parquet
files directly is outside the guard.  The guard prevents accidents, not sabotage.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


class HoldoutViolation(RuntimeError):
    """A research read reached into the sealed holdout period."""


class HoldoutAlreadySpent(RuntimeError):
    """The holdout has already been unlocked once."""


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class HoldoutLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def is_spent(self) -> bool:
        return any(e["event"] in ("unlock", "reopen") for e in self.events())

    def unlock(
        self, purpose: str, fingerprint: str | None = None, allow_reopen: bool = False
    ) -> dict:
        if not purpose.strip():
            raise ValueError("state the purpose of opening the holdout")
        if self.is_spent() and not allow_reopen:
            raise HoldoutAlreadySpent(
                "the holdout was already unlocked (see the ledger); a second look is a new, "
                "contaminated use -- pass allow_reopen=True to record it as such"
            )
        event = {
            "event": "reopen" if self.is_spent() else "unlock",
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "purpose": purpose,
            "git_commit": _git_commit(),
            "data_fingerprint": fingerprint,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(event) + "\n")
        return event
