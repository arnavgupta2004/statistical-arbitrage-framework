"""Structured data-quality findings.

Cleaning never edits data silently: every drop, repair or flag is an ``Issue`` that lands in the
fetch log / validation report, so the number of observations removed is itself a reported result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

import pandas as pd


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Issue:
    ticker: str
    code: str
    severity: Severity
    count: int
    detail: str
    first_date: pd.Timestamp | None = None
    last_date: pd.Timestamp | None = None


ISSUE_COLUMNS = ["ticker", "code", "severity", "count", "first_date", "last_date", "detail"]


def issues_to_frame(issues: list[Issue]) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame(columns=ISSUE_COLUMNS)
    rows = [{**asdict(i), "severity": i.severity.value} for i in issues]
    return pd.DataFrame(rows)[ISSUE_COLUMNS]


def summarize(issues: list[Issue]) -> pd.DataFrame:
    """Counts per (code, severity), the headline data-quality table."""
    df = issues_to_frame(issues)
    if df.empty:
        return pd.DataFrame(columns=["code", "severity", "tickers", "observations"])
    return (
        df.groupby(["code", "severity"])
        .agg(tickers=("ticker", "nunique"), observations=("count", "sum"))
        .reset_index()
        .sort_values(["severity", "observations"], ascending=[True, False])
        .reset_index(drop=True)
    )
