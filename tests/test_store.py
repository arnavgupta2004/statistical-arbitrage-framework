from __future__ import annotations

import pandas as pd
import pytest

from statarb.data.sources.synthetic import SyntheticSource
from statarb.data.storage.store import ParquetStore, frame_hash

from .conftest import make_world, ts


@pytest.fixture
def history():
    raw = SyntheticSource(make_world()).fetch("AAA", ts("2015-01-01"), ts("2016-12-30"))
    return raw.prices.drop(columns="adj_close"), raw.actions


@pytest.fixture
def store(tmp_path):
    with ParquetStore(tmp_path / "s") as s:
        yield s


def test_roundtrip_is_exact(store, history):
    prices, actions = history
    store.write_history("AAA", prices, actions)
    pd.testing.assert_frame_equal(store.read_prices("AAA"), prices, check_freq=False)
    pd.testing.assert_frame_equal(store.read_actions("AAA"), actions.reset_index(drop=True))


def test_missing_ticker_returns_none_and_empty_actions(store):
    assert store.read_prices("NOPE") is None
    assert store.read_actions("NOPE").empty and not store.has_history("NOPE")


def test_ticker_with_special_characters(store, history):
    prices, actions = history
    store.write_history("^VIX", prices, actions)
    store.write_history("BRK-B", prices, actions)
    assert store.has_history("^VIX") and store.has_history("BRK-B")
    assert store.read_prices("^VIX").shape == prices.shape


def test_write_is_atomic_and_leaves_no_temp_files(store, history):
    prices, actions = history
    store.write_history("AAA", prices, actions)
    store.write_history("AAA", prices.iloc[:100], actions)  # overwrite in place
    assert len(store.read_prices("AAA")) == 100
    assert not list(store.root.rglob("*.tmp"))


def test_content_hash_is_stable_and_sensitive(history):
    prices, actions = history
    assert frame_hash(prices, actions) == frame_hash(prices.copy(), actions.copy())
    tweaked = prices.copy()
    tweaked.iloc[10, tweaked.columns.get_loc("close")] += 1e-9
    assert frame_hash(tweaked, actions) != frame_hash(prices, actions)
    assert frame_hash(prices, actions.iloc[:-1]) != frame_hash(prices, actions)


def test_manifest_upsert_merges_fields_and_roundtrips_types(store):
    assert store.get_entry("AAA") is None
    store.upsert_entry(
        "AAA",
        source="synthetic",
        status="ok",
        first_date=ts("2015-01-02"),
        last_date=ts("2016-12-30"),
        n_rows=500,
        content_hash="abc",
    )
    store.upsert_entry("AAA", message="hello", last_attempt_at=ts("2020-01-01 12:00"))
    e = store.get_entry("AAA")
    assert e["status"] == "ok" and e["n_rows"] == 500 and e["message"] == "hello"
    assert e["last_date"] == ts("2016-12-30") and e["last_attempt_at"] == ts("2020-01-01 12:00")
    assert e["last_full_refresh_at"] is None
    assert list(store.manifest()["ticker"]) == ["AAA"]


def test_sql_views_span_all_tickers(store, history):
    prices, actions = history
    store.write_history("AAA", prices, actions)
    store.write_history("BBB", prices, actions)
    out = store.sql("SELECT ticker, count(*) AS n FROM prices GROUP BY ticker ORDER BY ticker")
    assert out["ticker"].tolist() == ["AAA", "BBB"] and (out["n"] == len(prices)).all()
    assert store.sql("SELECT count(*) AS n FROM actions WHERE kind = 'split'")["n"][0] == 2


def test_fetch_log_is_append_only(store):
    store.log(ts("2020-01-01"), "AAA", "full", "ok", 10, "x")
    store.log(ts("2020-01-02"), "AAA", "incremental", "ok", 1, "y")
    assert store.fetch_log()["mode"].tolist() == ["full", "incremental"]


def test_ingest_issues_are_persisted(store):
    from statarb.data.cleaning.issues import Issue, Severity

    store.log_issues(
        ts("2020-01-01"),
        [
            Issue(
                "AAA",
                "DUPLICATE_DATES",
                Severity.WARNING,
                2,
                "dup",
                ts("2019-01-02"),
                ts("2019-01-03"),
            ),
            Issue("BBB", "ADJ_CLOSE_MISMATCH", Severity.WARNING, 1, "x"),
        ],
    )
    df = store.ingest_issues()
    assert df["code"].tolist() == ["DUPLICATE_DATES", "ADJ_CLOSE_MISMATCH"]
    assert df["severity"].tolist() == ["warning", "warning"] and df["count"].tolist() == [2, 1]
    assert df["first_date"].iloc[0] == ts("2019-01-02") and pd.isna(df["first_date"].iloc[1])
