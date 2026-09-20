"""Typed, validated configuration for the data layer.

Every threshold that changes what data a research result is computed on lives here rather than
as a literal in the code, so that it can be recorded in the experiment registry (Stage 8+) and
reproduced.  Unknown keys are rejected: a mistyped threshold must fail loudly, not be ignored.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RefreshConfig(_Strict):
    overlap_days: int = Field(
        10, ge=1, description="calendar days re-fetched to detect restatements"
    )
    full_refresh_after_days: int = Field(
        90, ge=0, description="force a full re-download of a ticker this old (0 = never)"
    )
    retry_no_data_days: int = Field(
        30, ge=0, description="min days between retries of empty tickers"
    )
    request_pause_s: float = Field(0.25, ge=0, description="pause between vendor requests")
    max_retries: int = Field(3, ge=0)
    restatement_tol: float = Field(
        1e-4,
        gt=0,
        description="relative close difference on overlapping rows that counts as a restatement",
    )
    finalize_buffer_hours: float = Field(
        2.0, ge=0, description="hours after the closing bell before a session is treated as final"
    )


class UniverseConfig(_Strict):
    mode: Literal["static", "membership_file", "sp500_wikipedia"] = "sp500_wikipedia"
    tickers: list[str] = Field(default_factory=list, description="mode=static")
    membership_file: str | None = Field(
        None, description="mode=membership_file: CSV ticker,start,end[,sector]"
    )
    aliases_file: str | None = Field(None, description="CSV old,new mapping for renamed tickers")
    snapshot: str = Field(
        "latest", description="mode=sp500_wikipedia: 'latest' or a YYYY-MM-DD snapshot"
    )
    expected_size: int = Field(
        500, description="target index size for the reconstruction size check"
    )
    size_tolerance: float = Field(
        0.03, gt=0, description="relative size deviation tolerated as 'reliable'"
    )
    min_volume_share: float = Field(
        0.95, gt=0, le=1, description="identity check: share of observed bars with volume > 0"
    )


class ValidationConfig(_Strict):
    max_abs_log_return: float = Field(0.5, gt=0, description="|log return| above this is flagged")
    stale_run_length: int = Field(
        5, ge=2, description="identical closes in a row that count as stale"
    )
    max_gap_sessions: int = Field(
        5, ge=1, description="run of missing sessions that escalates to an error"
    )
    split_ratio_tol: float = Field(
        0.03,
        gt=0,
        description="|log(jump * ratio)| tolerated when matching a jump to a split ratio",
    )
    max_dividend_yield: float = Field(
        0.10, gt=0, description="a 'dividend' above this share of the prior close is non-ordinary"
    )
    adj_close_tol: float = Field(
        5e-3, gt=0, description="daily-return gap vs vendor adj-close tolerated"
    )
    ohlc_tol: float = Field(1e-6, ge=0)


class AlignmentConfig(_Strict):
    min_coverage: float = Field(
        0.95, gt=0, le=1, description="observed share of a window for eligibility"
    )


class DataConfig(_Strict):
    store_dir: str = "var/store"
    source: Literal["yahoo", "csv", "synthetic"] = "yahoo"
    source_options: dict = Field(default_factory=dict)
    start: date = date(2005, 1, 3)
    calendar: str = "XNYS"
    refresh: RefreshConfig = RefreshConfig()
    universe: UniverseConfig = UniverseConfig()
    validation: ValidationConfig = ValidationConfig()
    alignment: AlignmentConfig = AlignmentConfig()

    @property
    def store_path(self) -> Path:
        return Path(self.store_dir)


def load_config(path: str | Path) -> DataConfig:
    """Load a YAML config; relative paths inside it resolve against the current directory."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return DataConfig.model_validate(raw)
