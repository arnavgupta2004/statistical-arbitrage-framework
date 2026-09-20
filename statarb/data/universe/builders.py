"""Configurable universe construction.

Every mode returns a ``Universe``: point-in-time membership, a sector map, machine-readable
provenance and a list of caveats.  ``caveats`` is not decoration -- results tables must print it,
and no mode may be described as free of survivorship bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from statarb.config import UniverseConfig
from statarb.data.universe.membership import Membership, reconstruct_from_changes
from statarb.data.universe.sp500 import load_snapshot

UNKNOWN_SECTOR = "Unknown"


@dataclass
class Universe:
    membership: Membership
    sectors: pd.Series  # ticker -> GICS sector ("Unknown" where no classification is available)
    provenance: dict = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)
    change_log: pd.DataFrame | None = None  # effective-date add/remove log, when reconstructed

    def members_on(self, date) -> list[str]:
        return self.membership.members_on(date)

    def tickers_ever(self, start, end) -> list[str]:
        return self.membership.ever_members(start, end)

    def sector_of(self, tickers) -> pd.Series:
        return self.sectors.reindex(list(tickers)).fillna(UNKNOWN_SECTOR)


def load_aliases(path: str | Path | None) -> dict[str, str]:
    """``old,new`` CSV of renamed tickers; the whole history of ``old`` is mapped onto ``new``."""
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path, comment="#")
    return dict(zip(df["old"].str.upper(), df["new"].str.upper(), strict=True))


def _static(cfg: UniverseConfig) -> Universe:
    tickers = sorted({t.upper() for t in cfg.tickers})
    if not tickers:
        raise ValueError("universe.mode=static needs universe.tickers")
    iv = pd.DataFrame({"ticker": tickers, "start": pd.NaT, "end": pd.NaT})
    return Universe(
        Membership(iv),
        pd.Series(UNKNOWN_SECTOR, index=tickers, name="sector"),
        provenance={"mode": "static", "n_tickers": len(tickers)},
        caveats=[
            "STATIC LIST: every ticker is treated as a member for all dates. If the list was "
            "chosen using knowledge of who survived or performed well, results are biased by "
            "hindsight (survivorship / selection bias) by an unknown amount."
        ],
    )


def _from_file(cfg: UniverseConfig) -> Universe:
    if not cfg.membership_file:
        raise ValueError("universe.mode=membership_file needs universe.membership_file")
    df = pd.read_csv(cfg.membership_file, parse_dates=["start", "end"])
    df["ticker"] = df["ticker"].str.upper()
    sectors = (
        df.dropna(subset=["sector"]).drop_duplicates("ticker").set_index("ticker")["sector"]
        if "sector" in df.columns
        else pd.Series(dtype=object)
    )
    return Universe(
        Membership(df[["ticker", "start", "end"]]),
        sectors.reindex(sorted(df["ticker"].unique())).fillna(UNKNOWN_SECTOR).rename("sector"),
        provenance={
            "mode": "membership_file",
            "file": str(cfg.membership_file),
            "n_intervals": len(df),
        },
        caveats=[
            "USER-SUPPLIED MEMBERSHIP: point-in-time quality is only as good as the file; "
            "the pipeline cannot verify completeness."
        ],
    )


def _sp500(cfg: UniverseConfig, universe_dir: Path) -> Universe:
    constituents, changes, label = load_snapshot(universe_dir, cfg.snapshot)
    aliases = load_aliases(cfg.aliases_file)
    current = sorted({aliases.get(t, t) for t in constituents["ticker"]})
    log = changes.copy()
    for col in ("added", "removed"):
        log[col] = log[col].map(lambda t: aliases.get(t, t) if isinstance(t, str) else t)
    membership, anomalies = reconstruct_from_changes(current, log)

    sectors = (
        constituents.assign(ticker=constituents["ticker"].map(lambda t: aliases.get(t, t)))
        .drop_duplicates("ticker")
        .set_index("ticker")["sector"]
    )
    all_tickers = sorted(membership.intervals["ticker"].unique())
    return Universe(
        membership,
        sectors.reindex(all_tickers).fillna(UNKNOWN_SECTOR).rename("sector"),
        provenance={
            "mode": "sp500_wikipedia",
            "snapshot": label,
            "n_current": len(current),
            "n_changes": len(changes),
            "earliest_change": str(changes["date"].min().date()),
            "latest_change": str(changes["date"].max().date()),
            "n_anomalies": len(anomalies),
            "n_aliases_applied": len(aliases),
            "n_tickers_ever": len(all_tickers),
        },
        caveats=[
            "RECONSTRUCTED membership: walked backwards from the snapshot using a "
            "volunteer-maintained change log. The log is sparse before ~2010, so membership is "
            "only trustworthy from the 'reliable_from' date reported by the size check.",
            "Tickers that were acquired, went bankrupt or were renamed often have no free price "
            "history; the resulting price universe is survivor-tilted. Coverage by date is "
            "reported by `survivorship_report`; the return of the missing names is unobservable.",
            "GICS sectors are today's classification applied to all dates (look-ahead in sector "
            "labels); delisted names have sector 'Unknown'.",
        ],
        anomalies=anomalies,
        change_log=changes,
    )


def build_universe(cfg: UniverseConfig, store_root: str | Path) -> Universe:
    if cfg.mode == "static":
        return _static(cfg)
    if cfg.mode == "membership_file":
        return _from_file(cfg)
    if cfg.mode == "sp500_wikipedia":
        return _sp500(cfg, Path(store_root) / "universe")
    raise ValueError(f"unknown universe mode {cfg.mode!r}")
