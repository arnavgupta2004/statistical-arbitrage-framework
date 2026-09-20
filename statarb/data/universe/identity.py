"""Does the stored price series belong to the company that was in the index?

Vendors key history by *current* symbol.  A ticker reused after a company left the index (or
delisted) then returns an unrelated instrument's prices -- often an illiquid stub whose stale,
zero-volume history sits under the name of a former index member.  Counting that as "we have prices
for it" would overstate coverage and, worse, feed a stale microcap into a pair screen as if it were
a large cap.

Test, per membership interval ``[start, end)``: while a name is in the S&P 500 it trades every
session with volume.  We therefore require, over the sessions of the interval that fall inside the
stored window, that (almost) every observed bar has positive volume.

==========  ==============================================================================
verdict     meaning
==========  ==============================================================================
ok          liquid trading observed for (part of) the interval
no_prices   the vendor has no rows inside the interval
suspect     rows exist but too many have zero volume: probably another company's history
==========  ==============================================================================

``obs_share`` (observed / expected sessions) is reported alongside: a name that delisted
mid-interval is legitimately partial, so it is *not* a verdict by itself.  This is a heuristic
with false negatives (a reused symbol that happens to be liquid passes): it bounds, and does not
remove, the contamination.
"""

from __future__ import annotations

import pandas as pd

from statarb.data.universe.builders import Universe

IDENTITY_COLUMNS = [
    "ticker",
    "start",
    "end",
    "n_expected",
    "n_obs",
    "obs_share",
    "volume_share",
    "verdict",
]


def check_identity(
    universe: Universe,
    read_prices,
    sessions: pd.DatetimeIndex,
    start,
    end,
    min_volume_share: float = 0.95,
) -> pd.DataFrame:
    """One row per membership interval.  ``read_prices(ticker) -> DataFrame | None``."""
    lo_all, hi_all = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    cache: dict[str, pd.DataFrame | None] = {}
    for t, s, e in universe.membership.intervals[["ticker", "start", "end"]].itertuples(
        index=False
    ):
        lo = max(lo_all, s) if pd.notna(s) else lo_all
        hi = min(hi_all, e - pd.Timedelta(days=1)) if pd.notna(e) else hi_all
        expected = sessions[(sessions >= lo) & (sessions <= hi)]
        if not len(expected):
            continue
        if t not in cache:
            cache[t] = read_prices(t)
        p = cache[t]
        sub = p.loc[lo:hi] if p is not None else None
        n_obs = 0 if sub is None else len(sub)
        vol_share = float((sub["volume"] > 0).mean()) if n_obs else float("nan")
        if n_obs == 0:
            verdict = "no_prices"
        elif vol_share < min_volume_share:
            verdict = "suspect"
        else:
            verdict = "ok"
        rows.append((t, s, e, len(expected), n_obs, n_obs / len(expected), vol_share, verdict))
    return pd.DataFrame(rows, columns=IDENTITY_COLUMNS)
