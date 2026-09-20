"""Point-in-time index membership as half-open intervals ``[start, end)``.

``start`` NaT means "member since before the record begins"; ``end`` NaT means "still a member".
``members_on(d)`` therefore uses only information whose effective date is <= d: a name added
effective ``d`` is a member on ``d`` (its first trading day in the index), a name removed
effective ``d`` is not.  Effective dates are used instead of announcement dates, which is the
conservative reading: announcements precede effectiveness by days, so no announced-but-not-yet-
effective information is ever used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Membership:
    intervals: pd.DataFrame  # columns: ticker, start, end

    def __post_init__(self) -> None:
        df = self.intervals.copy()
        for c in ("start", "end"):
            df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
        self.intervals = df.sort_values(["ticker", "start"], na_position="first").reset_index(
            drop=True
        )

    def _active(self, date: pd.Timestamp) -> pd.Series:
        d = pd.Timestamp(date)
        s, e = self.intervals["start"], self.intervals["end"]
        return (s.isna() | (s <= d)) & (e.isna() | (d < e))

    def members_on(self, date) -> list[str]:
        return sorted(self.intervals.loc[self._active(date), "ticker"].unique())

    def ever_members(self, start, end) -> list[str]:
        """Tickers that were members on at least one day of [start, end]."""
        s, e = self.intervals["start"], self.intervals["end"]
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        overlaps = (s.isna() | (s <= hi)) & (e.isna() | (e > lo))
        return sorted(self.intervals.loc[overlaps, "ticker"].unique())

    def size(self, dates) -> pd.Series:
        idx = pd.DatetimeIndex(dates)
        return pd.Series([len(self.members_on(d)) for d in idx], index=idx, name="size")

    def matrix(self, dates) -> pd.DataFrame:
        """Boolean (date x ticker) membership matrix, for masking panels point-in-time."""
        idx = pd.DatetimeIndex(dates)
        tickers = sorted(self.intervals["ticker"].unique())
        out = pd.DataFrame(False, index=idx, columns=tickers)
        for t, g in self.intervals.groupby("ticker"):
            for s, e in zip(g["start"], g["end"], strict=True):
                m = np.ones(len(idx), dtype=bool)
                if pd.notna(s):
                    m &= idx >= s
                if pd.notna(e):
                    m &= idx < e
                out[t] = out[t].to_numpy() | m
        return out


def reconstruct_from_changes(
    current: list[str], changes: pd.DataFrame
) -> tuple[Membership, list[dict]]:
    """Rebuild historical membership by walking the change log *backwards* from today's members.

    ``changes`` has columns ``date`` (effective), ``added``, ``removed`` (ticker or None; one row
    per swap).  Stepping backwards across a change dated ``d``: every ticker added on ``d`` was not
    a member before ``d`` (its interval starts at ``d``); every ticker removed on ``d`` was a member
    before ``d`` (a new interval ending at ``d`` opens).

    Returns the membership and a list of *anomalies* -- log entries inconsistent with the state
    they should apply to (e.g. "added" a name that is not a member afterwards).  Anomalies mean the
    log is incomplete or the ticker was renamed; they are reported, never hidden.

    Limitation: exact only if the log is complete.  A missed removal makes earlier membership too
    small; a missed addition makes it too large (see ``survivorship.reconstruction_size_check``).
    """
    active: dict[str, dict] = {t: {"start": pd.NaT, "end": pd.NaT} for t in current}
    closed: list[dict] = []
    anomalies: list[dict] = []

    log = changes.copy()
    log["date"] = pd.to_datetime(log["date"])
    for d, grp in log.sort_values("date", ascending=False, kind="stable").groupby(
        "date", sort=False
    ):
        for a in grp["added"].dropna():
            if a in active:
                closed.append({"ticker": a, "start": d, "end": active[a]["end"]})
                del active[a]
            else:
                anomalies.append(
                    {"date": d, "ticker": a, "kind": "added_but_not_member_afterwards"}
                )
        for r in grp["removed"].dropna():
            if r in active:
                anomalies.append({"date": d, "ticker": r, "kind": "removed_but_member_afterwards"})
            else:
                active[r] = {"start": pd.NaT, "end": d}
    for t, iv in active.items():
        closed.append({"ticker": t, "start": iv["start"], "end": iv["end"]})
    return Membership(pd.DataFrame(closed, columns=["ticker", "start", "end"])), anomalies
