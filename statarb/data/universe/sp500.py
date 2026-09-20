"""S&P 500 constituents and change log scraped from Wikipedia (CC BY-SA), snapshotted locally.

Sources
-------
* ``List of S&P 500 companies``            -> today's constituents with GICS sector / sub-industry
* ``Historical components of the S&P 500`` -> effective-date change log (added / removed)

Parsing is a pure function of HTML so it is unit-tested offline; fetching is separate and every
fetch is stored as a dated Parquet snapshot so a universe can be rebuilt bit-for-bit later even
though the live page keeps changing.

Known weaknesses (all reported by ``survivorship``): the change log is dense only from ~2010 and
sparse before ~2007; tickers in the log are those of the day (renames such as FB -> META need the
alias table); GICS sectors are *today's* classification (a look-ahead for earlier dates); and
companies removed after bankruptcy or acquisition often have no free price history at all.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import requests

USER_AGENT = "statarb-research/0.1 (non-commercial research; python-requests)"
CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"


def clean_ticker(x) -> str | None:
    """Yahoo-style symbol: strip footnote markers/whitespace, upper-case, ``.`` -> ``-``."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = re.sub(r"\[.*?\]", "", str(x)).strip().upper()
    if not s or s in {"NAN", "—", "-"}:
        return None
    return s.replace(".", "-")


def parse_constituents(html: str) -> pd.DataFrame:
    """Columns: ticker, name, sector, sub_industry, date_added."""
    for t in pd.read_html(io.StringIO(html)):
        cols = [str(c) for c in t.columns]
        if "Symbol" in cols and "GICS Sector" in cols:
            df = pd.DataFrame(
                {
                    "ticker": t["Symbol"].map(clean_ticker),
                    "name": t["Security"],
                    "sector": t["GICS Sector"],
                    "sub_industry": t["GICS Sub-Industry"] if "GICS Sub-Industry" in cols else None,
                    "date_added": pd.to_datetime(t["Date added"], errors="coerce")
                    if "Date added" in cols
                    else pd.NaT,
                }
            )
            return df.dropna(subset=["ticker"]).reset_index(drop=True)
    raise ValueError("no constituents table found")


def parse_changes(html: str) -> pd.DataFrame:
    """Columns: date (effective), added, removed, added_name, removed_name, reason."""
    for t in pd.read_html(io.StringIO(html)):
        if not isinstance(t.columns, pd.MultiIndex) or "Effective Date" not in {
            c[0] for c in t.columns
        }:
            continue
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(t.iloc[:, 0], errors="coerce", format="mixed"),
                "added": t.iloc[:, 1].map(clean_ticker),
                "added_name": t.iloc[:, 2],
                "removed": t.iloc[:, 3].map(clean_ticker),
                "removed_name": t.iloc[:, 4],
                "reason": t.iloc[:, 5] if t.shape[1] > 5 else None,
            }
        )
        # the header is repeated as the first data row on the live page; unparseable dates drop it
        df = df.dropna(subset=["date"])
        return (
            df[df["added"].notna() | df["removed"].notna()]
            .sort_values("date")
            .reset_index(drop=True)
        )
    raise ValueError("no change-log table found")


def fetch_html(url: str, timeout: float = 30.0) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r.text


def fetch_snapshot(universe_dir: Path, today: pd.Timestamp | None = None) -> str:
    """Download, parse and store today's snapshot; returns its date label (YYYY-MM-DD)."""
    label = (today or pd.Timestamp.now()).strftime("%Y-%m-%d")
    constituents = parse_constituents(fetch_html(CURRENT_URL))
    changes = parse_changes(fetch_html(CHANGES_URL))
    universe_dir.mkdir(parents=True, exist_ok=True)
    constituents.to_parquet(universe_dir / f"sp500_constituents_{label}.parquet", index=False)
    changes.to_parquet(universe_dir / f"sp500_changes_{label}.parquet", index=False)
    return label


def list_snapshots(universe_dir: Path) -> list[str]:
    return sorted(
        p.stem.removeprefix("sp500_constituents_")
        for p in universe_dir.glob("sp500_constituents_*.parquet")
    )


def load_snapshot(
    universe_dir: Path, which: str = "latest"
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    labels = list_snapshots(universe_dir)
    if not labels:
        raise FileNotFoundError(
            f"no S&P 500 snapshot in {universe_dir}; run the 'universe --fetch' command"
        )
    label = labels[-1] if which == "latest" else which
    if label not in labels:
        raise FileNotFoundError(f"snapshot {label} not found; available: {labels}")
    return (
        pd.read_parquet(universe_dir / f"sp500_constituents_{label}.parquet"),
        pd.read_parquet(universe_dir / f"sp500_changes_{label}.parquet"),
        label,
    )
