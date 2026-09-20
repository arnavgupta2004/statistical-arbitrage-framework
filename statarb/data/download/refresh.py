"""Incremental refresh of the local store.

Why not "just append new rows"
------------------------------
Vendors restate history.  A split after the last download rescales every earlier close and
volume, so appending only new rows would splice two different bases together and manufacture a
fake price jump on the join.  Each refresh therefore re-fetches an *overlap window* of recent
rows and compares it with what is stored; if any close moved by more than ``restatement_tol`` the
vendor has revised history and the ticker is re-downloaded in full.  Independently, a full
re-download is forced every ``full_refresh_after_days`` to catch silent restatements (for example
back-filled dividends) that do not touch the overlap window.

Ticker plan
-----------
::

    no manifest entry / force_full        -> FULL
    status no_data, retried recently      -> SKIP
    last full refresh too old             -> FULL
    stored through the last final session -> SKIP (up to date)
    otherwise                             -> INCREMENTAL (overlap window; escalates to FULL)

Only *finalised* sessions are stored (``TradingCalendar.last_complete_session``): a bar fetched
while the market is open or just closed is provisional and would be revised on the next refresh.
A failure for one ticker is logged and never aborts the run or corrupts stored data.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import pandas as pd

from statarb.config import DataConfig
from statarb.data.cleaning.calendar import TradingCalendar
from statarb.data.cleaning.issues import Issue, Severity
from statarb.data.cleaning.sanitize import sanitize_history
from statarb.data.cleaning.validation import reconcile_vendor_adjustment
from statarb.data.sources.base import (
    ACTION_COLUMNS,
    DataSource,
    RawHistory,
    SourceError,
    TransientSourceError,
    empty_actions,
)
from statarb.data.storage.store import ParquetStore

log = logging.getLogger(__name__)


@dataclass
class TickerResult:
    ticker: str
    action: str  # full | incremental | skipped | no_data | error
    rows_written: int = 0
    reason: str = ""
    issues: list[Issue] = field(default_factory=list)


@dataclass
class RefreshReport:
    end: pd.Timestamp
    results: list[TickerResult] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.action] = out.get(r.action, 0) + 1
        return out

    @property
    def issues(self) -> list[Issue]:
        return [i for r in self.results for i in r.issues]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ticker": r.ticker, "action": r.action, "rows": r.rows_written, "reason": r.reason}
                for r in self.results
            ]
        )


class Refresher:
    def __init__(
        self,
        store: ParquetStore,
        source: DataSource,
        cfg: DataConfig,
        calendar: TradingCalendar | None = None,
        now_fn: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.store, self.source, self.cfg = store, source, cfg
        self.calendar = calendar or TradingCalendar(cfg.calendar)
        self.now_fn, self.sleep = now_fn, sleep

    # ---- public --------------------------------------------------------------------------
    def refresh(
        self, tickers: Iterable[str], *, force_full: bool = False, end: pd.Timestamp | None = None
    ) -> RefreshReport:
        now = self.now_fn()
        end = (
            pd.Timestamp(end)
            if end is not None
            else self.calendar.last_complete_session(now, self.cfg.refresh.finalize_buffer_hours)
        )
        report = RefreshReport(end=end)
        for ticker in sorted(set(tickers)):
            try:
                result = self._refresh_one(ticker, end, force_full, now.tz_localize(None))
            except Exception as exc:  # never let one symbol abort a 500-symbol run
                log.warning("refresh failed for %s: %s", ticker, exc)
                result = TickerResult(ticker, "error", reason=f"{type(exc).__name__}: {exc}")
                self.store.upsert_entry(
                    ticker,
                    status=self._status_on_error(ticker),
                    last_attempt_at=now.tz_localize(None),
                    message=result.reason,
                )
            self.store.log_issues(now.tz_localize(None), result.issues)
            self.store.log(
                now.tz_localize(None),
                ticker,
                result.action,
                "error" if result.action == "error" else "ok",
                result.rows_written,
                result.reason,
            )
            report.results.append(result)
        return report

    # ---- planning --------------------------------------------------------------------------
    def _plan(
        self, entry: dict | None, end: pd.Timestamp, force_full: bool, now: pd.Timestamp
    ) -> tuple[str, str]:
        rc = self.cfg.refresh
        if force_full:
            return "full", "forced"
        if entry is None:
            return "full", "new ticker"
        if entry["status"] == "no_data":
            last = entry["last_attempt_at"]
            if last is not None and (now - last) < pd.Timedelta(days=rc.retry_no_data_days):
                return "skip", "no data upstream; retried recently"
            return "full", "retry after no_data"
        if entry["last_date"] is None:  # earlier attempt failed before storing anything
            return "full", "previous attempt failed"
        full_at = entry["last_full_refresh_at"]
        if rc.full_refresh_after_days and (
            full_at is None or (now - full_at) >= pd.Timedelta(days=rc.full_refresh_after_days)
        ):
            return "full", "periodic full refresh"
        if entry["last_date"] >= end:
            return "skip", "up to date"
        # A history that ends long before ``end`` (delisted / acquired) returns nothing new on every
        # probe; back off instead of asking the vendor about it on each run.
        empty_at = entry["last_empty_at"]
        if (
            empty_at is not None
            and entry["last_date"] < end - pd.Timedelta(days=rc.overlap_days)
            and (now - empty_at) < pd.Timedelta(days=rc.retry_no_data_days)
        ):
            return "skip", "history ended upstream; probed recently"
        return "incremental", "new sessions"

    def _refresh_one(
        self, ticker: str, end: pd.Timestamp, force_full: bool, now: pd.Timestamp
    ) -> TickerResult:
        entry = self.store.get_entry(ticker)
        mode, reason = self._plan(entry, end, force_full, now)
        if mode == "skip":
            return TickerResult(ticker, "skipped", reason=reason)

        start = pd.Timestamp(self.cfg.start)
        if mode == "incremental":
            stored = self.store.read_prices(ticker)
            window_start = max(
                start, entry["last_date"] - pd.Timedelta(days=self.cfg.refresh.overlap_days)
            )
            raw = self._fetch(ticker, window_start, end)
            prices, actions, issues = self._clean(raw)
            if (
                prices.empty
            ):  # no new sessions (or vendor silent): never overwrite stored rows with nothing
                self.store.upsert_entry(
                    ticker,
                    last_attempt_at=now,
                    last_empty_at=self._empty_marker(entry["last_date"], end, now),
                    message="incremental: no rows returned",
                )
                return TickerResult(ticker, "incremental", 0, "no rows returned", issues)
            n_bad, detail = self._restated(stored, prices)
            if n_bad == 0:
                return self._merge_and_write(
                    ticker, stored, prices, actions, window_start, issues, now, end
                )
            reason = f"restatement detected ({detail}); full re-download"

        raw = self._fetch(ticker, start, end)
        prices, actions, issues = self._clean(raw)
        if prices.empty:
            if entry is not None and entry["last_date"] is not None:
                # history exists locally: an empty full response is a vendor glitch, not a delisting
                msg = "full re-download returned no rows; stored history kept"
                self.store.upsert_entry(ticker, last_attempt_at=now, message=msg)
                return TickerResult(ticker, "error", reason=msg, issues=issues)
            self.store.upsert_entry(
                ticker,
                source=self.source.name,
                status="no_data",
                last_attempt_at=now,
                message="vendor returned no rows",
            )
            return TickerResult(ticker, "no_data", reason="vendor returned no rows", issues=issues)
        old = self.store.read_prices(ticker)
        if old is not None and len(old):
            n_bad, detail = self._restated(old, prices, whole_history=True)
            if n_bad:
                issues.append(
                    Issue(
                        ticker,
                        "VENDOR_RESTATEMENT",
                        Severity.INFO,
                        n_bad,
                        f"full refetch differs from stored history: {detail}",
                    )
                )
        h = self.store.write_history(ticker, prices, actions)
        self._record(ticker, prices, h, now, full=True, message=reason)
        return TickerResult(ticker, "full", len(prices), reason, issues)

    # ---- helpers ---------------------------------------------------------------------------
    def _fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> RawHistory:
        attempt = 0
        while True:
            try:
                raw = self.source.fetch(ticker, start, end)
                self.sleep(self.cfg.refresh.request_pause_s)
                return raw
            except TransientSourceError:
                attempt += 1
                if attempt > self.cfg.refresh.max_retries:
                    raise
                self.sleep(min(2.0**attempt, 30.0))
            except SourceError:
                raise

    def _clean(self, raw: RawHistory) -> tuple[pd.DataFrame, pd.DataFrame, list[Issue]]:
        prices, actions, issues = sanitize_history(raw.ticker, raw.prices, raw.actions)
        issues += reconcile_vendor_adjustment(raw.ticker, prices, actions, self.cfg.validation)
        return prices, actions, issues

    def _restated(
        self, stored: pd.DataFrame | None, new: pd.DataFrame, whole_history: bool = False
    ) -> tuple[int, str]:
        """Stored closes the vendor now reports differently (beyond ``restatement_tol``)."""
        if stored is None or stored.empty or new.empty:
            return 0, "nothing to compare"
        lo = new.index.min()
        overlap = (
            stored.index.intersection(new.index)
            if whole_history
            else stored.loc[lo:].index.intersection(new.index)
        )
        if not len(overlap):
            return 0, "no overlapping rows"
        rel = (new.loc[overlap, "close"] / stored.loc[overlap, "close"] - 1.0).abs()
        n_bad = int((rel > self.cfg.refresh.restatement_tol).sum())
        return n_bad, f"{n_bad}/{len(overlap)} rows changed, max rel diff {rel.max():.2e}"

    def _merge_and_write(
        self, ticker, stored, prices, actions, window_start, issues, now, end
    ) -> TickerResult:
        """Union of stored and new bars (new wins on a shared date); stored rows are never dropped.

        Actions inside the overlap window are replaced by the response (a revised dividend amount
        must not be counted twice); actions before it are kept as stored.
        """
        merged = pd.concat([stored, prices]) if stored is not None else prices
        merged = merged.sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        old_actions = self.store.read_actions(ticker)
        parts = [
            f
            for f in (
                old_actions[old_actions["date"] < window_start],
                actions[actions["date"] >= window_start],
            )
            if len(f)
        ]
        merged_a = (
            pd.concat(parts)
            .drop_duplicates(["date", "kind", "value"])
            .sort_values(["date", "kind"])
            .reset_index(drop=True)
            if parts
            else empty_actions()
        )[ACTION_COLUMNS]
        n_new = len(merged) - (len(stored) if stored is not None else 0)
        h = self.store.write_history(ticker, merged, merged_a)
        self._record(ticker, merged, h, now, full=False, message="incremental")
        if n_new <= 0:  # the probe found nothing newer than what we already held
            marker = self._empty_marker(merged.index.max(), end, now)
            self.store.upsert_entry(ticker, last_empty_at=marker)
        return TickerResult(ticker, "incremental", max(n_new, 0), "new sessions", issues)

    def _empty_marker(self, last_date: pd.Timestamp, end: pd.Timestamp, now: pd.Timestamp):
        """When to remember an empty probe: only if the stored history is clearly not current."""
        stale = last_date < end - pd.Timedelta(days=self.cfg.refresh.overlap_days)
        return now if stale else None

    def _record(self, ticker, prices, content_hash, now, *, full: bool, message: str) -> None:
        fields = dict(
            source=self.source.name,
            status="ok",
            first_date=prices.index.min(),
            last_date=prices.index.max(),
            n_rows=len(prices),
            content_hash=content_hash,
            last_attempt_at=now,
            last_success_at=now,
            last_empty_at=None,
            message=message,
        )
        if full:
            fields["last_full_refresh_at"] = now
        self.store.upsert_entry(ticker, **fields)

    def _status_on_error(self, ticker: str) -> str:
        e = self.store.get_entry(ticker)
        return "ok" if e and e["last_date"] is not None else "error"
