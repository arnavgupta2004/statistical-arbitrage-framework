"""Non-mutating audits of a stored (or freshly downloaded) price history.

Each check returns ``Issue`` records; none of them edits data.  What is checked, and why it matters
for statistical arbitrage:

============================  ==========================================================
Code                          Why it matters
============================  ==========================================================
MISSING_SESSIONS              gaps create fake multi-day returns and break pair alignment
CALENDAR_MISMATCH             rows on non-sessions (or vendor "today" bars) are not tradeable
FUTURE_DATES                  rows after the last finalised session are provisional
STALE_PRICE                   a frozen close looks like a perfectly mean-reverting spread
EXTREME_RETURN                data errors / unrecorded events dominate a mean-reversion signal
SPLIT_UNADJUSTED              a split left in the series is a fake +/-75% "reversion"
POSSIBLE_UNRECORDED_SPLIT     same, but the vendor never listed the split
OHLC_INCONSISTENT             bad bars (high < close ...) corrupt any range-based cost proxy
ZERO_VOLUME                   untradeable day; ADV and impact estimates degrade
ADJ_CLOSE_MISMATCH            our dividend adjustment disagrees with the vendor's
============================  ==========================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.config import ValidationConfig
from statarb.data.cleaning.corporate_actions import (
    assign_dividends_to_rows,
    dividends_of,
    splits_of,
)
from statarb.data.cleaning.issues import Issue, Severity

# ratios (new/old shares) that real splits take; used to recognise an unrecorded split from a jump
COMMON_SPLIT_RATIOS = (2, 3, 4, 5, 6, 8, 10, 15, 20, 1.5, 2 / 3, 0.5, 0.25, 0.2, 0.125, 0.1, 0.05)


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def audit_history(
    ticker: str,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    cfg: ValidationConfig,
    last_final_session: pd.Timestamp | None = None,
) -> list[Issue]:
    """Run every audit on one ticker.  ``sessions`` must cover [first, last] of ``prices``."""
    issues: list[Issue] = []
    if prices.empty:
        return [Issue(ticker, "NO_DATA", Severity.ERROR, 0, "no price rows")]
    idx = prices.index
    first, last = idx.min(), idx.max()

    # --- calendar --------------------------------------------------------------------------
    expected = sessions[(sessions >= first) & (sessions <= last)]
    missing = expected.difference(idx)
    if len(missing):
        pos = expected.isin(missing)
        longest = _longest_run(pos)
        sev = Severity.ERROR if longest > cfg.max_gap_sessions else Severity.WARNING
        issues.append(
            Issue(
                ticker,
                "MISSING_SESSIONS",
                sev,
                len(missing),
                f"{len(missing)}/{len(expected)} sessions absent; longest run {longest}",
                missing.min(),
                missing.max(),
            )
        )
    off_calendar = idx.difference(sessions)
    if len(off_calendar):
        issues.append(
            Issue(
                ticker,
                "CALENDAR_MISMATCH",
                Severity.ERROR,
                len(off_calendar),
                "rows on dates that are not trading sessions",
                off_calendar.min(),
                off_calendar.max(),
            )
        )
    if last_final_session is not None and last > last_final_session:
        n = int((idx > last_final_session).sum())
        issues.append(
            Issue(
                ticker,
                "FUTURE_DATES",
                Severity.ERROR,
                n,
                "rows after the last finalised session (provisional data)",
                idx[idx > last_final_session].min(),
                last,
            )
        )

    close = prices["close"]
    # --- stale prices ----------------------------------------------------------------------
    same = (close.diff() == 0).to_numpy()
    n_stale_rows = 0
    run = 0
    for v in same:
        run = run + 1 if v else 0
        if run + 1 == cfg.stale_run_length:
            n_stale_rows += cfg.stale_run_length
        elif run + 1 > cfg.stale_run_length:
            n_stale_rows += 1
    if n_stale_rows:
        issues.append(
            Issue(
                ticker,
                "STALE_PRICE",
                Severity.WARNING,
                n_stale_rows,
                f"runs of >= {cfg.stale_run_length} identical closes",
            )
        )

    # --- returns: extremes and splits -----------------------------------------------------
    ratio = (close / close.shift(1)).iloc[1:]
    logr = np.log(ratio)
    split_rows = _split_row_positions(idx, actions)  # positions of the first bar on/after ex-date
    near_split = np.zeros(len(idx), dtype=bool)
    for p in split_rows:
        near_split[max(p - 1, 0) : p + 2] = True
    near_split = near_split[1:]

    splits = splits_of(actions)
    for d, k in zip(splits["date"], splits["value"], strict=True):
        p = idx.searchsorted(d)
        if 0 < p < len(idx) and abs(k - 1) > 0.2:
            q = close.iloc[p] / close.iloc[p - 1]
            if abs(np.log(q * k)) < cfg.split_ratio_tol:
                issues.append(
                    Issue(
                        ticker,
                        "SPLIT_UNADJUSTED",
                        Severity.ERROR,
                        1,
                        f"close moved x{q:.4f} on split ratio {k:g}: history not split-adjusted",
                        idx[p],
                        idx[p],
                    )
                )

    big = (logr.abs() > cfg.max_abs_log_return).to_numpy() & ~near_split
    if big.any():
        worst = logr[big].abs().idxmax()
        issues.append(
            Issue(
                ticker,
                "EXTREME_RETURN",
                Severity.WARNING,
                int(big.sum()),
                f"|log return| > {cfg.max_abs_log_return}; "
                f"worst {logr[worst]:+.3f} on {worst.date()}",
                logr.index[big].min(),
                logr.index[big].max(),
            )
        )

    qv = ratio.to_numpy()
    jump = (np.abs(logr.to_numpy()) > np.log(1.4)) & ~near_split
    unrecorded = 0
    for i in np.flatnonzero(jump):
        if any(abs(np.log(qv[i] * k)) < cfg.split_ratio_tol for k in COMMON_SPLIT_RATIOS):
            unrecorded += 1
    if unrecorded:
        issues.append(
            Issue(
                ticker,
                "POSSIBLE_UNRECORDED_SPLIT",
                Severity.WARNING,
                unrecorded,
                "jump matches a common split ratio but no split is recorded nearby",
            )
        )

    # --- non-ordinary distributions ---------------------------------------------------------
    # Spin-offs, special dividends and cash mergers appear as huge "dividends". Vendors treat them
    # inconsistently (a spin-off can be booked as a dividend *and* baked into the adjusted close),
    # so total returns around them are not trustworthy and pair windows should mask them.
    divs = dividends_of(actions)
    if len(divs):
        prev_close = close.shift(1).reindex(idx)
        pos = idx.searchsorted(divs["date"].to_numpy())
        ok = (pos > 0) & (pos < len(idx))
        yields = divs["value"].to_numpy()[ok] / prev_close.to_numpy()[pos[ok] - 1]
        big_div = yields > cfg.max_dividend_yield
        if big_div.any():
            when = divs["date"].to_numpy()[ok][big_div]
            issues.append(
                Issue(
                    ticker,
                    "LARGE_DIVIDEND",
                    Severity.WARNING,
                    int(big_div.sum()),
                    f"dividend > {cfg.max_dividend_yield:.0%} of prior close "
                    f"(max {yields.max():.1%}): spin-off / special / merger cash; "
                    "total returns around it are unreliable",
                    pd.Timestamp(when.min()),
                    pd.Timestamp(when.max()),
                )
            )

    # --- bar sanity -----------------------------------------------------------------------
    o, h, lo = prices["open"], prices["high"], prices["low"]
    tol = cfg.ohlc_tol
    bad_bar = (h < np.maximum(np.maximum(o, close), lo) * (1 - tol)) | (
        lo > np.minimum(np.minimum(o, close), h) * (1 + tol)
    )
    if bad_bar.any():
        issues.append(
            Issue(
                ticker,
                "OHLC_INCONSISTENT",
                Severity.WARNING,
                int(bad_bar.sum()),
                "high < max(open, close) or low > min(open, close)",
                idx[bad_bar.to_numpy()].min(),
                idx[bad_bar.to_numpy()].max(),
            )
        )
    zero_vol = (prices["volume"] == 0).to_numpy()
    if zero_vol.any():
        issues.append(
            Issue(
                ticker,
                "ZERO_VOLUME",
                Severity.INFO,
                int(zero_vol.sum()),
                "sessions with zero volume",
            )
        )
    return issues


def _split_row_positions(idx: pd.DatetimeIndex, actions: pd.DataFrame) -> list[int]:
    s = splits_of(actions)
    if not len(s):
        return []
    return [int(p) for p in idx.searchsorted(s["date"].to_numpy()) if p < len(idx)]


def reconcile_vendor_adjustment(
    ticker: str, prices: pd.DataFrame, actions: pd.DataFrame, cfg: ValidationConfig
) -> list[Issue]:
    """Compare our dividend-adjusted daily returns with the vendor's ``adj_close`` returns.

    Only consecutive rows *within the same download* are comparable (``adj_close`` is rescaled
    retroactively), which is why this runs on a fresh response and never on stored history.
    """
    if "adj_close" not in prices.columns or len(prices) < 3:
        return []
    close = prices["close"]
    div = assign_dividends_to_rows(prices.index, dividends_of(actions))
    ours = (close + div) / close.shift(1) - 1.0
    theirs = prices["adj_close"] / prices["adj_close"].shift(1) - 1.0
    diff = (ours - theirs).abs().dropna()
    bad = diff > cfg.adj_close_tol
    if not bad.any():
        return []
    return [
        Issue(
            ticker,
            "ADJ_CLOSE_MISMATCH",
            Severity.WARNING,
            int(bad.sum()),
            f"our adjusted return differs from vendor adj_close by up to {diff.max():.4f}",
            diff.index[bad].min(),
            diff.index[bad].max(),
        )
    ]


def compare_sources(a: pd.DataFrame, b: pd.DataFrame, tol: float = 5e-3) -> dict[str, float]:
    """Cross-check two vendors' closes on their common dates (e.g. Yahoo vs a manual Stooq file).

    Returns overlap size, share of dates whose closes differ by more than ``tol`` (relative), and
    the maximum relative difference.  Agreement of two independent scrapes is evidence against a
    vendor-specific error; disagreement is only a lead, not a verdict on which one is right.
    """
    common = a.index.intersection(b.index)
    if not len(common):
        return {"overlap": 0, "share_differing": float("nan"), "max_rel_diff": float("nan")}
    rel = (a.loc[common, "close"] / b.loc[common, "close"] - 1.0).abs()
    return {
        "overlap": int(len(common)),
        "share_differing": float((rel > tol).mean()),
        "max_rel_diff": float(rel.max()),
    }
