"""Quantifying (not eliminating) survivorship bias in the constructed universe.

Three diagnostics, each answering a different question:

1. ``reconstruction_size_check`` -- is the rebuilt membership internally plausible?  An index
   of ~500 names should reconstruct to ~500 members at every date.  A drift means the change log is
   missing entries, and marks the date before which the reconstruction should not be trusted.
2. ``naive_overlap`` -- how wrong would the *naive* backtest be?  If one used today's constituents
   for all history, only the share of each date's true members that still exist today would be
   in the universe; the rest (removed, acquired, bankrupt names) would be silently absent.
3. ``log_density_check`` -- is the change log *complete*?  The size check cannot see a swap that is
   missing from the log entirely (it removes one name and adds one, so the size is unchanged while
   the membership is wrong).  Turnover is roughly stable year to year, so a year with far fewer
   recorded changes than the recent norm has an incomplete log.  Each omitted swap replaces a
   non-survivor by a current member, so the reconstruction *itself* becomes survivor-biased there.
4. ``coverage`` -- how much of the true point-in-time universe do we have *prices* for?  This is
   the remaining bias after the best free data: names with no price history are lost, and they are
   disproportionately the failures.
"""

from __future__ import annotations

import pandas as pd

from statarb.data.universe.builders import Universe


def reconstruction_size_check(
    universe: Universe, dates, expected: int = 500, tol: float = 0.03
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Sampled membership size vs the expected index size, and the earliest 'reliable' date.

    ``reliable_from`` is the earliest sampled date such that *every* later sampled date is within
    ``tol`` (relative) of ``expected``; ``None`` if even the latest date fails.
    """
    sizes = universe.membership.size(dates)
    df = pd.DataFrame({"size": sizes, "deviation": sizes / expected - 1.0})
    ok = df["deviation"].abs() <= tol
    if not ok.iloc[-1]:
        return df, None
    bad = ok[~ok]
    reliable = ok.index[0] if bad.empty else ok.index[ok.index > bad.index[-1]][0]
    return df, reliable


def log_density_check(
    universe: Universe, min_ratio: float = 0.6, reference_years: int = 10
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Recorded changes per calendar year vs the median of the last ``reference_years`` full years.

    A year is *sparse* if it has fewer than ``min_ratio`` x the reference.  Returns the table and
    the first Jan-1 from which every later full year is dense (``None`` if the log is unusable).
    """
    log = universe.change_log
    if log is None or log.empty:
        return pd.DataFrame(columns=["changes", "ratio", "sparse"]), None
    years = pd.DatetimeIndex(log["date"]).year
    counts = pd.Series(1, index=years).groupby(level=0).sum()
    last_full = int(counts.index.max()) - 1  # the latest year is usually partial
    counts = counts.reindex(range(int(counts.index.min()), last_full + 1), fill_value=0)
    reference = float(counts.iloc[-reference_years:].median())
    df = pd.DataFrame({"changes": counts, "ratio": counts / reference})
    df["sparse"] = df["ratio"] < min_ratio
    dense_from = None
    if not df["sparse"].iloc[-1]:
        sparse_years = df.index[df["sparse"]]
        first = (int(sparse_years.max()) + 1) if len(sparse_years) else int(df.index.min())
        dense_from = pd.Timestamp(year=first, month=1, day=1)
    return df, dense_from


def naive_overlap(universe: Universe, dates, today: pd.Timestamp) -> pd.DataFrame:
    """Share of each date's true members that are still members on ``today``."""
    now = set(universe.members_on(today))
    rows = []
    for d in pd.DatetimeIndex(dates):
        members = set(universe.members_on(d))
        rows.append(
            {
                "date": d,
                "n_members": len(members),
                "n_still_member": len(members & now),
                "naive_overlap": len(members & now) / len(members) if members else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def coverage(
    universe: Universe, manifest: pd.DataFrame, dates, identity: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Share of each date's members with stored prices spanning that date.

    With ``identity`` (``identity.check_identity``), a member whose interval was judged
    ``suspect`` (price history belongs to a different company) or ``no_prices`` is *not* counted,
    so reused symbols do not inflate coverage.
    """
    have = manifest[(manifest["status"] == "ok") & manifest["first_date"].notna()].set_index(
        "ticker"
    )
    bad = pd.DataFrame(columns=["ticker", "start", "end"])
    if identity is not None:
        bad = identity.loc[
            identity["verdict"].isin(["suspect", "no_prices"]), ["ticker", "start", "end"]
        ]
    rows = []
    for d in pd.DatetimeIndex(dates):
        members = universe.members_on(d)
        excluded = set()
        if len(bad):
            active = (bad["start"].isna() | (bad["start"] <= d)) & (
                bad["end"].isna() | (d < bad["end"])
            )
            excluded = set(bad.loc[active, "ticker"])
        with_data = [
            t
            for t in members
            if t not in excluded
            and t in have.index
            and have.at[t, "first_date"] <= d + pd.Timedelta(days=7)
            and have.at[t, "last_date"] >= d
        ]
        rows.append(
            {
                "date": d,
                "n_members": len(members),
                "n_with_prices": len(with_data),
                "coverage": len(with_data) / len(members) if members else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def survivorship_report(
    universe: Universe,
    manifest: pd.DataFrame,
    dates,
    today: pd.Timestamp,
    expected: int = 500,
    tol: float = 0.03,
    identity: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """One table (per sampled date) plus a summary dict of headline numbers."""
    size, size_from = reconstruction_size_check(universe, dates, expected, tol)
    density, density_from = log_density_check(universe)
    candidates = [d for d in (size_from, density_from) if d is not None]
    # trustworthy only where *both* checks pass; if either has no passing region, there is none
    both = len(candidates) == (2 if universe.change_log is not None else 1)
    reliable_from = max(candidates) if candidates and both else None
    tbl = size.join(
        naive_overlap(universe, dates, today)[["n_still_member", "naive_overlap"]]
    ).join(coverage(universe, manifest, dates, identity)[["n_with_prices", "coverage"]])
    region = tbl if reliable_from is None else tbl.loc[reliable_from:]
    summary = {
        "reliable_from": None if reliable_from is None else str(reliable_from.date()),
        "reliable_from_size_check": None if size_from is None else str(size_from.date()),
        "reliable_from_log_density": None if density_from is None else str(density_from.date()),
        "sparse_log_years": [
            int(y)
            for y in density.index[density["sparse"]]
            if y >= pd.DatetimeIndex(dates).min().year
        ],
        "mean_naive_overlap_reliable_region": float(region["naive_overlap"].mean()),
        "min_naive_overlap_reliable_region": float(region["naive_overlap"].min()),
        "mean_price_coverage_reliable_region": float(region["coverage"].mean()),
        "min_price_coverage_reliable_region": float(region["coverage"].min()),
        "n_reconstruction_anomalies": len(universe.anomalies),
    }
    if identity is not None:
        summary["identity_verdicts"] = identity["verdict"].value_counts().to_dict()
    return tbl, summary
