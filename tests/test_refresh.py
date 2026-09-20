"""Incremental refresh: correctness of merging, restatement handling, failure isolation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.data.cleaning.issues import Severity
from statarb.data.download.refresh import Refresher
from statarb.data.sources.base import (
    RawHistory,
    SourceError,
    TransientSourceError,
    empty_actions,
    empty_prices,
)
from statarb.data.sources.synthetic import SyntheticSource

from .conftest import SPLIT_DATE, make_cfg, make_world, ts

NOW = pd.Timestamp("2016-12-31 23:00", tz="UTC")  # Saturday: last final session is 2016-12-30


def refresher(pipe, now=NOW, sleeps=None):
    return Refresher(
        pipe.store,
        pipe.source,
        pipe.cfg,
        pipe.calendar,
        now_fn=lambda: now,
        sleep=(lambda s: sleeps.append(s)) if sleeps is not None else (lambda s: None),
    )


def stored(pipe, t):
    return pipe.store.read_prices(t), pipe.store.read_actions(t)


def test_first_refresh_downloads_full_history_and_records_the_manifest(make_pipeline):
    pipe = make_pipeline()
    rep = refresher(pipe).refresh(["AAA", "BBB", "CCC", "DDD"])
    assert rep.counts() == {"full": 4}
    m = pipe.store.manifest().set_index("ticker")
    assert m.loc["AAA", "first_date"] == ts("2015-01-02") and m.loc["AAA", "last_date"] == ts(
        "2016-12-30"
    )
    assert m.loc["CCC", "first_date"] == ts(
        "2015-09-01"
    )  # late listing: history starts where it starts
    assert m.loc["DDD", "last_date"] == ts("2016-02-26")  # delisted: history ends where it ends
    assert (m["status"] == "ok").all() and m["content_hash"].notna().all()
    assert pipe.source.calls[0][1] == ts("2015-01-02")  # full history starts at cfg.start


def test_second_refresh_with_nothing_new_makes_no_vendor_calls(make_pipeline):
    pipe = make_pipeline()
    r = refresher(pipe)
    r.refresh(["AAA", "BBB"])
    n = len(pipe.source.calls)
    rep = r.refresh(["AAA", "BBB"])
    assert rep.counts() == {"skipped": 2} and len(pipe.source.calls) == n


def test_incremental_result_equals_a_fresh_full_download(make_pipeline):
    """The invariant that matters: refreshing in steps must give exactly what one big download gives."""
    inc = make_pipeline(vendor_date="2015-12-31", subdir="inc", full_refresh_after_days=0)
    refresher(inc, now=pd.Timestamp("2016-01-02 23:00", tz="UTC")).refresh(["BBB"])
    inc.source.vendor_date = ts("2016-12-30")  # a year of new sessions arrive (dividends included)
    rep = refresher(inc).refresh(["BBB"])
    assert rep.counts() == {"incremental": 1}
    # only the overlap window is re-downloaded, not the full history
    assert inc.source.calls[-1][1] > ts("2015-12-01")

    full = make_pipeline(vendor_date="2016-12-30", subdir="full")
    refresher(full).refresh(["BBB"])
    (p_inc, a_inc), (p_full, a_full) = stored(inc, "BBB"), stored(full, "BBB")
    pd.testing.assert_frame_equal(p_inc, p_full, check_freq=False)
    pd.testing.assert_frame_equal(a_inc, a_full)
    assert inc.store.get_entry("BBB")["content_hash"] == full.store.get_entry("BBB")["content_hash"]


def test_split_between_downloads_triggers_a_full_redownload_and_no_fake_jump(make_pipeline):
    pipe = make_pipeline(
        vendor_date="2016-03-11", subdir="a", full_refresh_after_days=0
    )  # AAA's 2-for-1 is on 2016-03-15
    refresher(pipe, now=pd.Timestamp("2016-03-12 23:00", tz="UTC")).refresh(["AAA"])
    before = stored(pipe, "AAA")[0]
    pipe.source.vendor_date = ts("2016-12-30")
    n_calls = len(pipe.source.calls)
    rep = refresher(pipe).refresh(["AAA"])
    assert rep.results[0].action == "full" and "restatement" in rep.results[0].reason
    assert len(pipe.source.calls) == n_calls + 2  # the overlap probe, then the full download

    after = stored(pipe, "AAA")[0]
    # history was restated: the old closes are now half what they were
    np.testing.assert_allclose(after["close"].loc[before.index] / before["close"], 0.5)
    # ... and, crucially, there is no splice: the series is continuous across the split date
    logret = np.log(after["close"]).diff().abs()
    assert logret.max() < 0.15
    fresh = make_pipeline(vendor_date="2016-12-30", subdir="b")
    refresher(fresh).refresh(["AAA"])
    pd.testing.assert_frame_equal(after, stored(fresh, "AAA")[0], check_freq=False)


def test_naive_append_would_have_spliced_two_bases(make_pipeline):
    """Documents the failure mode the overlap check prevents (so the check is not dead code)."""
    old = SyntheticSource(make_world(), vendor_date="2016-03-11").fetch(
        "AAA", ts("2015-01-01"), ts("2016-03-11")
    )
    new = SyntheticSource(make_world(), vendor_date="2016-12-30").fetch(
        "AAA", ts("2016-03-14"), ts("2016-12-30")
    )
    naive = pd.concat([old.prices, new.prices])
    assert np.log(naive["close"]).diff().abs().max() > 0.6  # fake ~-50% crash at the join


def test_periodic_full_refresh_is_forced(make_pipeline):
    pipe = make_pipeline(full_refresh_after_days=90)
    refresher(pipe, now=pd.Timestamp("2016-12-31 23:00", tz="UTC")).refresh(["BBB"])
    rep = refresher(pipe, now=pd.Timestamp("2017-04-15 23:00", tz="UTC")).refresh(["BBB"])
    assert rep.results[0].action == "full" and rep.results[0].reason == "periodic full refresh"
    pipe2 = make_pipeline(full_refresh_after_days=0, subdir="never")
    refresher(pipe2).refresh(["BBB"])
    assert refresher(pipe2, now=pd.Timestamp("2018-01-01 23:00", tz="UTC")).refresh(
        ["BBB"], end=ts("2016-12-30")
    ).counts() == {"skipped": 1}


def test_provisional_session_is_never_stored(make_pipeline):
    pipe = make_pipeline()
    # 2016-06-10 15:00 UTC = 11:00 New York on a trading day: the session has not even closed
    refresher(pipe, now=pd.Timestamp("2016-06-10 15:00", tz="UTC")).refresh(["BBB"])
    assert stored(pipe, "BBB")[0].index.max() == ts("2016-06-09")
    assert pipe.source.calls[0][2] == ts("2016-06-09")


def test_unknown_ticker_is_no_data_and_retried_only_after_the_backoff(make_pipeline):
    pipe = make_pipeline(retry_no_data_days=30)
    r = refresher(pipe)
    assert r.refresh(["ZZZ"]).counts() == {"no_data": 1}
    assert pipe.store.get_entry("ZZZ")["status"] == "no_data"
    n = len(pipe.source.calls)
    assert r.refresh(["ZZZ"]).counts() == {"skipped": 1} and len(pipe.source.calls) == n
    later = refresher(pipe, now=NOW + pd.Timedelta(days=31))
    assert later.refresh(["ZZZ"]).counts() == {"no_data": 1} and len(pipe.source.calls) == n + 1


class Flaky:
    """Wraps a source: raises on chosen tickers, or fails transiently N times before succeeding."""

    name = "flaky"

    def __init__(self, inner, permanent=(), transient=None, empty_after=None):
        self.inner, self.permanent, self.transient = inner, set(permanent), dict(transient or {})
        self.empty_after, self.n = empty_after, 0

    def fetch(self, ticker, start, end):
        self.n += 1
        if ticker in self.permanent:
            raise SourceError("boom")
        if self.transient.get(ticker, 0) > 0:
            self.transient[ticker] -= 1
            raise TransientSourceError("429")
        if self.empty_after is not None and self.n > self.empty_after:
            return RawHistory(ticker, empty_prices(), empty_actions(), "flaky", pd.Timestamp.now())
        return self.inner.fetch(ticker, start, end)


def test_one_bad_ticker_does_not_abort_the_run_or_touch_other_data(make_pipeline):
    pipe = make_pipeline()
    pipe.source = Flaky(pipe.source, permanent={"BBB"})
    rep = refresher(pipe).refresh(["AAA", "BBB", "CCC"])
    assert rep.counts() == {"full": 2, "error": 1}
    assert not pipe.store.has_history("BBB") and pipe.store.get_entry("BBB")["status"] == "error"
    assert "boom" in pipe.store.get_entry("BBB")["message"]
    log = pipe.store.fetch_log().set_index("ticker")
    assert log.loc["BBB", "status"] == "error" and log.loc["AAA", "status"] == "ok"


def test_transient_errors_are_retried_with_exponential_backoff(make_pipeline):
    pipe = make_pipeline(max_retries=3)
    pipe.source = Flaky(pipe.source, transient={"AAA": 2})
    sleeps: list[float] = []
    rep = refresher(pipe, sleeps=sleeps).refresh(["AAA"])
    assert rep.counts() == {"full": 1}
    assert [s for s in sleeps if s > 0] == [2.0, 4.0]  # 2**attempt back-off (request pause is 0)


def test_transient_errors_give_up_after_max_retries(make_pipeline):
    pipe = make_pipeline(max_retries=2)
    pipe.source = Flaky(pipe.source, transient={"AAA": 99})
    assert refresher(pipe).refresh(["AAA"]).counts() == {"error": 1}
    assert pipe.source.n == 3  # first try + 2 retries


def test_an_empty_incremental_response_never_deletes_stored_rows(make_pipeline):
    pipe = make_pipeline(vendor_date="2016-06-30", full_refresh_after_days=0)
    refresher(pipe, now=pd.Timestamp("2016-07-02 23:00", tz="UTC")).refresh(["BBB"])
    n_before = len(stored(pipe, "BBB")[0])
    pipe.source = Flaky(pipe.source, empty_after=0)  # vendor goes silent on the next call
    rep = refresher(pipe).refresh(["BBB"])
    assert rep.results[0].action == "incremental" and rep.results[0].rows_written == 0
    assert len(stored(pipe, "BBB")[0]) == n_before


def test_a_glitchy_empty_full_refetch_keeps_history_and_status(make_pipeline):
    pipe = make_pipeline()
    refresher(pipe).refresh(["BBB"])
    pipe.source = Flaky(pipe.source, empty_after=0)
    rep = refresher(pipe).refresh(["BBB"], force_full=True)
    assert rep.results[0].action == "error"
    assert pipe.store.get_entry("BBB")["status"] == "ok" and len(stored(pipe, "BBB")[0]) > 400


def test_vendor_restatement_on_full_refetch_is_reported(make_pipeline):
    pipe = make_pipeline(vendor_date="2016-03-11")
    refresher(pipe, now=pd.Timestamp("2016-03-12 23:00", tz="UTC")).refresh(["AAA"])
    pipe.source.vendor_date = ts("2016-12-30")
    rep = refresher(pipe).refresh(["AAA"], force_full=True)
    codes = {i.code: i for i in rep.results[0].issues}
    assert (
        codes["VENDOR_RESTATEMENT"].severity == Severity.INFO
        and codes["VENDOR_RESTATEMENT"].count > 0
    )


def test_config_start_bounds_the_history(tmp_path, calendar):
    cfg = make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"start": pd.Timestamp("2016-01-04").date()})
    from statarb.data.pipeline import DataPipeline

    pipe = DataPipeline(cfg, source=SyntheticSource(make_world()), calendar=calendar)
    try:
        refresher(pipe).refresh(["BBB"])
        assert stored(pipe, "BBB")[0].index.min() == ts("2016-01-04")
        assert SPLIT_DATE > "2016-01-04"
    finally:
        pipe.close()


class Trimmed:
    """A vendor that answers overlap probes with only the newest bars (drops the oldest ``n`` rows)."""

    name = "trimmed"

    def __init__(self, inner, n):
        self.inner, self.n = inner, n

    def fetch(self, ticker, start, end):
        h = self.inner.fetch(ticker, start, end)
        return RawHistory(ticker, h.prices.iloc[self.n :], h.actions, h.source, h.fetched_at)


def test_a_partial_overlap_response_never_deletes_stored_rows(make_pipeline):
    """If the vendor omits some overlap-window rows, what we already hold must survive the merge."""
    inc = make_pipeline(vendor_date="2016-06-30", subdir="inc", full_refresh_after_days=0)
    refresher(inc, now=pd.Timestamp("2016-07-02 23:00", tz="UTC")).refresh(["BBB"])
    inc.source.vendor_date = ts("2016-12-30")
    inc.source = Trimmed(inc.source, n=4)  # 4 of the ~7 overlap rows are missing from the response
    refresher(inc).refresh(["BBB"])

    full = make_pipeline(vendor_date="2016-12-30", subdir="full")
    refresher(full).refresh(["BBB"])
    pd.testing.assert_frame_equal(stored(inc, "BBB")[0], stored(full, "BBB")[0], check_freq=False)


class Duplicating:
    """A sloppy vendor that sends the first bar twice."""

    name = "dup"

    def __init__(self, inner):
        self.inner = inner

    def fetch(self, ticker, start, end):
        h = self.inner.fetch(ticker, start, end)
        return RawHistory(
            ticker, pd.concat([h.prices.iloc[:1], h.prices]), h.actions, h.source, h.fetched_at
        )


def test_repairs_made_at_ingest_are_recorded_not_silent(make_pipeline):
    pipe = make_pipeline()
    pipe.source = Duplicating(pipe.source)
    refresher(pipe).refresh(["BBB"])
    got = pipe.store.ingest_issues()
    assert got["ticker"].tolist() == ["BBB"] and got["code"].tolist() == ["DUPLICATE_DATES"]
    assert stored(pipe, "BBB")[0].index.is_unique  # ... and the stored data was actually repaired
