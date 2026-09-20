"""Ingest-time structural repair of a vendor response.

Only repairs that cannot change economics are applied, and each is reported:

* timestamps -> naive session dates, sorted ascending
* duplicate dates -> keep the last row (vendors re-send the corrected bar last)
* rows with a missing / non-positive close -> dropped (no price, no observation)
* negative volume -> NaN

Everything judgemental (extreme returns, stale prices, gaps, split residuals) is *flagged* by
``validation.audit_history`` and left in the data for the analyst to decide on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.data.cleaning.calendar import to_naive_dates
from statarb.data.cleaning.issues import Issue, Severity
from statarb.data.sources.base import ACTION_COLUMNS, PRICE_COLUMNS


def sanitize_history(
    ticker: str, prices: pd.DataFrame, actions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[Issue]]:
    issues: list[Issue] = []
    p = prices.copy()
    p.index = pd.DatetimeIndex(to_naive_dates(p.index), name="date")

    if not p.index.is_monotonic_increasing:
        issues.append(
            Issue(
                ticker,
                "UNSORTED_INDEX",
                Severity.WARNING,
                int((p.index[1:] < p.index[:-1]).sum()),
                "timestamps were not ascending; sorted",
            )
        )
        p = p.sort_index(kind="stable")

    dup = p.index.duplicated(keep="last")
    if dup.any():
        issues.append(
            Issue(
                ticker,
                "DUPLICATE_DATES",
                Severity.WARNING,
                int(dup.sum()),
                "duplicate timestamps; kept the last row per date",
                p.index[dup].min(),
                p.index[dup].max(),
            )
        )
        p = p[~dup]

    for col in PRICE_COLUMNS:
        if col not in p.columns:
            p[col] = np.nan
    bad_close = ~(p["close"] > 0)  # catches NaN, zero and negative
    if bad_close.any():
        issues.append(
            Issue(
                ticker,
                "INVALID_CLOSE",
                Severity.WARNING,
                int(bad_close.sum()),
                "close missing or non-positive; row dropped",
                p.index[bad_close].min(),
                p.index[bad_close].max(),
            )
        )
        p = p[~bad_close]

    neg_vol = p["volume"] < 0
    if neg_vol.any():
        issues.append(
            Issue(
                ticker,
                "NEGATIVE_VOLUME",
                Severity.WARNING,
                int(neg_vol.sum()),
                "negative volume set to NaN",
            )
        )
        p.loc[neg_vol, "volume"] = np.nan

    partial = p[["open", "high", "low"]].isna().any(axis=1)
    if partial.any():
        issues.append(
            Issue(
                ticker,
                "MISSING_OHL",
                Severity.INFO,
                int(partial.sum()),
                "open/high/low missing while close is present (kept)",
                p.index[partial].min(),
                p.index[partial].max(),
            )
        )

    a = actions.copy() if len(actions) else pd.DataFrame(columns=ACTION_COLUMNS)
    if len(a):
        a["date"] = to_naive_dates(a["date"])
        a["value"] = a["value"].astype("float64")
        bad = ~(a["value"] > 0) | ~a["kind"].isin(["dividend", "split"])
        if bad.any():
            issues.append(
                Issue(
                    ticker,
                    "INVALID_ACTION",
                    Severity.WARNING,
                    int(bad.sum()),
                    "corporate action with non-positive value or unknown kind; dropped",
                )
            )
            a = a[~bad]
        n0 = len(a)
        a = a.drop_duplicates(["date", "kind", "value"])
        if len(a) < n0:
            issues.append(
                Issue(
                    ticker,
                    "DUPLICATE_ACTIONS",
                    Severity.INFO,
                    n0 - len(a),
                    "duplicate actions dropped",
                )
            )
        a = a.sort_values(["date", "kind"]).reset_index(drop=True)
    return p, a, issues
