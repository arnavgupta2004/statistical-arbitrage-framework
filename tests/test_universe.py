"""Point-in-time universe: exact reconstruction on a known ground truth, parsing, survivorship."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from statarb.config import UniverseConfig
from statarb.data.universe.builders import UNKNOWN_SECTOR, Universe, build_universe, load_aliases
from statarb.data.universe.membership import Membership, reconstruct_from_changes
from statarb.data.universe.sp500 import (
    clean_ticker,
    fetch_snapshot,
    list_snapshots,
    load_snapshot,
    parse_changes,
    parse_constituents,
)
from statarb.data.universe.survivorship import (
    coverage,
    naive_overlap,
    reconstruction_size_check,
    survivorship_report,
)

from .conftest import ts


def log(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "added", "removed"]).assign(
        date=lambda d: pd.to_datetime(d["date"])
    )


# ---- membership semantics -----------------------------------------------------------------
def test_half_open_semantics_on_the_effective_date():
    m = Membership(
        pd.DataFrame(
            {
                "ticker": ["X", "Y"],
                "start": [ts("2020-03-02"), pd.NaT],
                "end": [pd.NaT, ts("2020-03-02")],
            }
        )
    )
    assert m.members_on("2020-03-01") == [
        "Y"
    ]  # X not yet added; Y still a member the day before removal
    assert m.members_on("2020-03-02") == [
        "X"
    ]  # added effective d is in on d; removed effective d is out on d


def test_reconstruction_by_hand():
    current = ["A", "B", "C"]
    changes = log(("2020-01-10", "C", "D"), ("2019-05-01", "B", "E"))
    m, anomalies = reconstruct_from_changes(current, changes)
    assert anomalies == []
    assert m.members_on("2021-01-01") == ["A", "B", "C"]
    assert m.members_on("2020-01-09") == ["A", "B", "D"]  # D leaves, C joins on 2020-01-10
    assert m.members_on("2019-04-30") == ["A", "D", "E"]  # before B replaced E
    assert m.ever_members("2019-01-01", "2019-12-31") == ["A", "B", "D", "E"]


def test_ticker_removed_and_later_readded_gets_two_intervals():
    # truth: {A,B,Z} -> (2018-06-01: A out, Q in) -> {B,Z,Q} -> (2021-05-01: A back in, Z out) -> {A,B,Q}
    m, anomalies = reconstruct_from_changes(
        ["A", "B", "Q"], log(("2021-05-01", "A", "Z"), ("2018-06-01", "Q", "A"))
    )
    assert anomalies == []
    assert m.members_on("2017-01-01") == ["A", "B", "Z"]
    assert m.members_on("2019-01-01") == ["B", "Q", "Z"]  # A is out during 2018-06 .. 2021-05
    assert m.members_on("2022-01-01") == ["A", "B", "Q"]
    assert (m.intervals["ticker"] == "A").sum() == 2


def test_inconsistent_log_entries_are_reported_as_anomalies_not_hidden():
    # the log claims ZZZ joined (but it is not a member afterwards) and A left (but A is a member today)
    m, anomalies = reconstruct_from_changes(["A", "B"], log(("2020-01-01", "ZZZ", "A")))
    assert sorted(a["kind"] for a in anomalies) == [
        "added_but_not_member_afterwards",
        "removed_but_member_afterwards",
    ]
    assert {a["ticker"] for a in anomalies} == {"ZZZ", "A"}
    assert m.members_on("2019-01-01") == [
        "A",
        "B",
    ]  # inconsistent entries are skipped, not guessed at


def test_several_swaps_on_one_effective_date():
    current = ["A", "B", "C", "D"]
    m, anomalies = reconstruct_from_changes(
        current, log(("2020-06-01", "C", "X"), ("2020-06-01", "D", "Y"))
    )
    assert anomalies == []
    assert m.members_on("2020-05-31") == ["A", "B", "X", "Y"]
    assert m.members_on("2020-06-01") == ["A", "B", "C", "D"]


def test_matrix_agrees_with_members_on():
    m, _ = reconstruct_from_changes(
        ["A", "B", "C"], log(("2020-01-10", "C", "D"), ("2019-05-01", "B", "E"))
    )
    dates = pd.date_range("2019-01-01", "2021-01-01", freq="7D")
    mat = m.matrix(dates)
    for d in dates[::9]:
        assert sorted(mat.columns[mat.loc[d]]) == m.members_on(d)
    assert (mat.sum(axis=1) == m.size(dates)).all()


@st.composite
def membership_worlds(draw):
    pool = [f"T{i}" for i in range(14)]
    members = set(draw(st.lists(st.sampled_from(pool), min_size=5, max_size=9, unique=True)))
    initial = set(members)
    n = draw(st.integers(1, 14))
    dates = pd.date_range("2010-01-04", periods=n, freq="30D")
    rows, truth = [], []
    for d in dates:
        out = draw(st.sampled_from(sorted(members)))
        cand = sorted(set(pool) - members)
        add = draw(st.sampled_from(cand))
        members = (members - {out}) | {add}
        rows.append((d, add, out))
        truth.append((d, frozenset(members)))
    return initial, rows, truth, frozenset(members)


@settings(max_examples=60, deadline=None)
@given(membership_worlds())
def test_reconstruction_is_exact_when_the_log_is_complete(world):
    """Property: forward-simulate a random index, then rebuild it backwards from only today's members."""
    initial, rows, truth, final = world
    m, anomalies = reconstruct_from_changes(sorted(final), log(*rows))
    assert anomalies == []
    assert set(m.members_on("2009-12-01")) == initial  # before the first change
    prev = initial
    for d, after in truth:
        assert set(m.members_on(d - pd.Timedelta(days=1))) == prev
        assert set(m.members_on(d)) == after
        prev = set(after)


# ---- ticker cleaning and HTML parsing --------------------------------------------------------
@pytest.mark.parametrize(
    "raw, want",
    [
        ("BRK.B", "BRK-B"),
        (" aapl[1] ", "AAPL"),
        (None, None),
        (float("nan"), None),
        ("", None),
        ("BF.B", "BF-B"),
    ],
)
def test_clean_ticker(raw, want):
    assert clean_ticker(raw) == want


CONSTITUENTS_HTML = """
<table class="wikitable"><thead><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>
<th>GICS Sub-Industry</th><th>Headquarters Location</th><th>Date added</th><th>CIK</th><th>Founded</th></tr></thead>
<tbody>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>Hardware</td><td>Cupertino</td><td>1982-11-30</td><td>1</td><td>1977</td></tr>
<tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>Multi-Sector</td><td>Omaha</td><td>2010-02-16</td><td>2</td><td>1839</td></tr>
</tbody></table>
<table><tr><td>noise</td></tr></table>
"""

CHANGES_HTML = """
<table class="wikitable"><thead>
<tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th rowspan="2">Reason</th><th rowspan="2">Refs</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr></thead>
<tbody>
<tr><td>Effective Date</td><td>Ticker</td><td>Security</td><td>Ticker</td><td>Security</td><td>Reason</td><td>Refs</td></tr>
<tr><td>August 18, 2026</td><td>RDDT</td><td>Reddit</td><td>AVB</td><td>AvalonBay</td><td>merger</td><td>[2]</td></tr>
<tr><td>June 9, 2022</td><td>META[1]</td><td>Meta</td><td></td><td></td><td>size</td><td></td></tr>
<tr><td>July 1, 2019</td><td></td><td></td><td>BF.B</td><td>Brown-Forman</td><td>delisted</td><td></td></tr>
</tbody></table>
"""


def test_parse_constituents():
    df = parse_constituents(CONSTITUENTS_HTML)
    assert df["ticker"].tolist() == ["AAPL", "BRK-B"]
    assert df["sector"].tolist() == ["Information Technology", "Financials"]
    assert df["date_added"].iloc[0] == ts("1982-11-30")


def test_parse_changes_skips_the_repeated_header_and_one_sided_rows_survive():
    df = parse_changes(CHANGES_HTML)
    assert df["date"].tolist() == [
        ts("2019-07-01"),
        ts("2022-06-09"),
        ts("2026-08-18"),
    ]  # sorted, header dropped
    assert df["added"].isna().tolist() == [True, False, False] and df[
        "added"
    ].dropna().tolist() == ["META", "RDDT"]
    assert df["removed"].isna().tolist() == [False, True, False] and df[
        "removed"
    ].dropna().tolist() == ["BF-B", "AVB"]


def test_parse_errors_on_wrong_page():
    with pytest.raises(ValueError):
        parse_constituents("<table><tr><td>x</td></tr></table>")
    with pytest.raises(ValueError):
        parse_changes("<table><tr><td>x</td></tr></table>")


def test_snapshot_roundtrip_with_fetch_mocked(tmp_path, monkeypatch):
    from statarb.data.universe import sp500

    pages = {sp500.CURRENT_URL: CONSTITUENTS_HTML, sp500.CHANGES_URL: CHANGES_HTML}
    monkeypatch.setattr(sp500, "fetch_html", lambda url, timeout=30.0: pages[url])
    label = fetch_snapshot(tmp_path / "u", today=ts("2026-09-20"))
    assert label == "2026-09-20" and list_snapshots(tmp_path / "u") == ["2026-09-20"]
    cons, changes, got = load_snapshot(tmp_path / "u")
    assert got == label and len(cons) == 2 and len(changes) == 3
    with pytest.raises(FileNotFoundError):
        load_snapshot(tmp_path / "u", "1999-01-01")
    with pytest.raises(FileNotFoundError):
        load_snapshot(tmp_path / "empty")


# ---- builders ---------------------------------------------------------------------------------
def test_static_universe_carries_a_survivorship_caveat():
    u = build_universe(UniverseConfig(mode="static", tickers=["aaa", "BBB"]), "unused")
    assert (
        u.members_on("1999-01-01") == ["AAA", "BBB"]
        and u.sector_of(["AAA"])["AAA"] == UNKNOWN_SECTOR
    )
    assert any("survivorship" in c.lower() for c in u.caveats)
    with pytest.raises(ValueError):
        build_universe(UniverseConfig(mode="static"), "unused")


def test_membership_file_universe(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text(
        "ticker,start,end,sector\nAAA,2010-01-01,2015-06-30,Tech\nBBB,,,Health\nAAA,2018-01-01,,Tech\n"
    )
    u = build_universe(UniverseConfig(mode="membership_file", membership_file=str(f)), tmp_path)
    assert u.members_on("2012-01-01") == ["AAA", "BBB"] and u.members_on("2016-01-01") == ["BBB"]
    assert u.members_on("2019-01-01") == ["AAA", "BBB"]
    assert u.sector_of(["AAA", "BBB"]).tolist() == ["Tech", "Health"]


def test_sp500_universe_applies_the_alias_table(tmp_path):
    udir = tmp_path / "universe"
    udir.mkdir()
    pd.DataFrame(
        {
            "ticker": ["META", "AAPL"],
            "name": ["Meta", "Apple"],
            "sector": ["Comm", "Tech"],
            "sub_industry": ["", ""],
            "date_added": pd.NaT,
        }
    ).to_parquet(udir / "sp500_constituents_2026-09-20.parquet")
    pd.DataFrame(
        {
            "date": [ts("2013-12-11")],
            "added": ["FB"],
            "removed": ["XYZ"],
            "added_name": [""],
            "removed_name": [""],
            "reason": [""],
        }
    ).to_parquet(udir / "sp500_changes_2026-09-20.parquet")
    aliases = tmp_path / "a.csv"
    aliases.write_text("# comment\nold,new\nFB,META\n")
    assert (
        load_aliases(aliases) == {"FB": "META"}
        and load_aliases(None) == {}
        and load_aliases("/nope") == {}
    )

    cfg = UniverseConfig(mode="sp500_wikipedia", aliases_file=str(aliases))
    u = build_universe(cfg, tmp_path)
    assert u.members_on("2013-12-10") == ["AAPL", "XYZ"]  # META (as FB) joined on the 11th
    assert u.members_on("2013-12-11") == ["AAPL", "META"] and u.anomalies == []
    assert u.sector_of(["META", "XYZ"]).tolist() == ["Comm", UNKNOWN_SECTOR]
    # without the alias the rename shows up as reconstruction anomalies instead of being hidden
    u2 = build_universe(UniverseConfig(mode="sp500_wikipedia"), tmp_path)
    assert len(u2.anomalies) == 1 and u2.provenance["n_anomalies"] == 1
    assert any("survivor" in c.lower() for c in u2.caveats)


# ---- survivorship diagnostics ----------------------------------------------------------------
def _universe_with_gap() -> Universe:
    """10 stable members, except 2010-2011 where the log 'forgot' three of them (7 members)."""
    rows = []
    for i in range(10):
        start = pd.NaT if i < 7 else ts("2012-01-01")
        end = ts("2016-01-01") if i == 0 else pd.NaT  # T0 leaves in 2016 (a non-survivor)
        rows.append({"ticker": f"T{i}", "start": start, "end": end})
    return Universe(Membership(pd.DataFrame(rows)), pd.Series(dtype=object))


def test_size_check_finds_the_reliable_from_date():
    u = _universe_with_gap()
    dates = pd.to_datetime(["2010-01-01", "2011-01-01", "2012-01-01", "2014-01-01", "2016-06-01"])
    df, reliable = reconstruction_size_check(u, dates, expected=10, tol=0.05)
    assert (
        df["size"].tolist() == [7, 7, 10, 10, 9] and reliable is None
    )  # last date is 9 -> 10% off

    df, reliable = reconstruction_size_check(u, dates, expected=10, tol=0.15)
    assert reliable == ts("2012-01-01")  # first date after which every sampled date is within 15%


def test_naive_overlap_measures_who_a_today_only_universe_would_miss():
    u = _universe_with_gap()
    out = naive_overlap(u, pd.to_datetime(["2014-01-01", "2016-06-01"]), today=ts("2016-06-01"))
    assert out.loc["2014-01-01", "n_members"] == 10 and out.loc["2014-01-01", "n_still_member"] == 9
    assert out.loc["2014-01-01", "naive_overlap"] == pytest.approx(
        0.9
    )  # the naive universe omits T0
    assert out.loc["2016-06-01", "naive_overlap"] == 1.0


def test_coverage_counts_only_names_with_prices_spanning_the_date():
    u = _universe_with_gap()
    manifest = pd.DataFrame(
        {
            "ticker": ["T0", "T1", "T2", "T3"],
            "status": ["ok", "ok", "no_data", "ok"],
            "first_date": [ts("2005-01-03"), ts("2005-01-03"), pd.NaT, ts("2015-01-02")],
            "last_date": [ts("2015-12-31"), ts("2020-01-01"), pd.NaT, ts("2020-01-01")],
        }
    )
    cov = coverage(u, manifest, pd.to_datetime(["2014-01-01"]))
    assert (
        cov.loc["2014-01-01", "n_with_prices"] == 2
    )  # T0,T1 yes; T2 no data; T3 starts 2015; rest absent
    assert cov.loc["2014-01-01", "coverage"] == pytest.approx(0.2)


def test_survivorship_report_summary_keys():
    u = _universe_with_gap()
    manifest = pd.DataFrame(
        {
            "ticker": ["T1"],
            "status": ["ok"],
            "first_date": [ts("2005-01-03")],
            "last_date": [ts("2020-01-01")],
        }
    )
    tbl, summary = survivorship_report(
        u,
        manifest,
        pd.to_datetime(["2012-01-01", "2014-01-01", "2016-06-01"]),
        today=ts("2016-06-01"),
        expected=10,
        tol=0.15,
    )
    assert (
        summary["reliable_from"] == "2012-01-01"
        and 0 < summary["mean_price_coverage_reliable_region"] < 1
    )
    assert list(tbl.columns) == [
        "size",
        "deviation",
        "n_still_member",
        "naive_overlap",
        "n_with_prices",
        "coverage",
    ]


# ---- change-log density: what the size check cannot see -------------------------------------
def _log_with_yearly_counts(counts: dict[int, int]) -> Universe:
    rows = []
    for year, n in counts.items():
        for k in range(n):
            rows.append(
                (
                    pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=k),
                    f"A{year}_{k}",
                    None,
                )
            )
    m = Membership(pd.DataFrame({"ticker": ["Z"], "start": [pd.NaT], "end": [pd.NaT]}))
    return Universe(m, pd.Series(dtype=object), change_log=log(*rows))


def test_density_check_finds_the_first_dense_year():
    from statarb.data.universe.survivorship import log_density_check

    counts = {2005: 1, 2006: 2, 2007: 11, 2008: 8, 2009: 13, 2010: 11} | {
        y: 20 for y in range(2011, 2027)
    }
    table, dense_from = log_density_check(_log_with_yearly_counts(counts))
    assert 2026 not in table.index  # the latest (partial) year is not judged
    assert table.loc[2010, "changes"] == 11 and bool(table.loc[2010, "sparse"])  # 11 < 0.6 * 20
    assert not bool(table.loc[2011, "sparse"])
    assert dense_from == pd.Timestamp("2011-01-01")


def test_size_check_alone_is_blind_to_a_missing_swap_but_density_is_not():
    """One swap (a removal AND an addition) absent from the log leaves the size at 10 but is wrong."""
    from statarb.data.universe.survivorship import reconstruction_size_check

    truth = [f"T{i}" for i in range(10)]
    # true history: T0 was in the index until 2015 and T9 joined in 2015; the log never recorded it
    rebuilt = Universe(
        Membership(pd.DataFrame({"ticker": truth, "start": pd.NaT, "end": pd.NaT})),
        pd.Series(dtype=object),
    )
    df, reliable = reconstruction_size_check(
        rebuilt, pd.to_datetime(["2010-01-01", "2020-01-01"]), 10, 0.05
    )
    assert (
        reliable == ts("2010-01-01") and (df["size"] == 10).all()
    )  # ...size check sees nothing wrong
    sparse = _log_with_yearly_counts({2010: 1, 2011: 1, 2012: 20, 2013: 20, 2014: 20, 2015: 20})
    from statarb.data.universe.survivorship import log_density_check

    assert log_density_check(sparse, reference_years=4)[1] == pd.Timestamp(
        "2012-01-01"
    )  # ...density does


def test_reliable_from_is_the_later_of_the_two_checks():
    counts = {2008: 5} | {y: 20 for y in range(2009, 2027)}
    u = _log_with_yearly_counts(counts)
    manifest = pd.DataFrame(columns=["ticker", "status", "first_date", "last_date"])
    dates = pd.to_datetime(["2006-01-01", "2010-01-01", "2020-01-01"])
    _, summary = survivorship_report(
        u, manifest, dates, today=ts("2020-01-01"), expected=1, tol=0.05
    )
    assert summary["reliable_from_size_check"] == "2006-01-01"
    assert summary["reliable_from_log_density"] == "2009-01-01"
    assert summary["reliable_from"] == "2009-01-01" and summary["sparse_log_years"] == [2008]
