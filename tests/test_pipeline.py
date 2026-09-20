"""End-to-end: universe -> download -> validate -> corporate actions -> align, and its causality."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from statarb.config import UniverseConfig
from statarb.data.cleaning.issues import Severity
from statarb.data.leakage import truncation_invariance
from statarb.data.pipeline import DataPipeline
from statarb.data.sources.synthetic import SyntheticSource

from .conftest import make_cfg, make_world, ts

ALL = ["AAA", "BBB", "CCC", "DDD"]
START = "2015-01-02"
PANEL_FIELDS = ["close", "open", "high", "low", "volume", "dividends", "ret", "observed", "growth"]


@pytest.fixture
def full(make_pipeline) -> DataPipeline:
    """Vendor at the end of the world: knows about the 2016-03-15 split."""
    pipe = make_pipeline(subdir="full")
    pipe.refresh()
    return pipe


def assert_panels_equal(a, b):
    for f in PANEL_FIELDS:
        pd.testing.assert_frame_equal(getattr(a, f), getattr(b, f), check_freq=False, obj=f)


@pytest.mark.parametrize("as_of", ["2016-02-26", "2016-03-14", "2016-03-15", "2016-08-01"])
def test_panel_as_of_equals_the_store_a_vendor_would_have_had_on_that_day(
    make_pipeline, full, as_of
):
    """The point-in-time contract, end to end.

    ``full`` was downloaded at the end of the world.  Asking it for a panel ending at ``as_of`` must
    give bit-identical values to a store that was *actually downloaded on* ``as_of`` -- around a
    split, with dividends, late listings and a delisting in the sample.
    """
    then = make_pipeline(vendor_date=as_of, subdir=f"then_{as_of}")
    then.refresh()
    assert_panels_equal(full.panel(ALL, START, as_of), then.panel(ALL, START, as_of))


def test_default_as_of_is_the_end_of_the_panel_and_levels_depend_on_it(full):
    early = full.panel(["AAA"], START, "2016-02-26")
    late = full.panel(["AAA"], START, "2016-02-26", as_of="2016-12-30")  # deliberately non-causal
    np.testing.assert_allclose(late.close["AAA"].dropna() / early.close["AAA"].dropna(), 0.5)
    # returns, the quantity research should consume, do not care:
    pd.testing.assert_frame_equal(early.ret, late.ret)
    # dollar volume is split-invariant
    pd.testing.assert_frame_equal(early.dollar_volume, late.dollar_volume)


def test_nothing_after_as_of_is_ever_visible(full):
    p = full.panel(ALL, START, "2016-12-30", as_of="2016-06-30")
    after = p.close.index > ts("2016-06-30")
    assert p.close.loc[after].isna().all().all() and not p.observed.loc[after].any().any()
    assert p.dividends.loc[after].sum().sum() == 0
    assert p.close.loc[~after].notna().any().any()


def test_returns_are_truncation_invariant_through_the_whole_read_path(full):
    """The generic detector applied to the pipeline itself (store read -> rebase -> align -> returns).

    Only *returns* are asserted invariant.  Price *levels* legitimately differ between a view ending
    before a split and one ending after it (the split rescales history), which is exactly why
    ``as_of`` exists; that contract is tested by the as-of-equivalence test above.
    """
    sessions = full.calendar.sessions(START, "2016-12-30")
    frame = pd.DataFrame({"session": np.arange(len(sessions), dtype=float)}, index=sessions)

    def ret_through_pipeline(f: pd.DataFrame) -> pd.DataFrame:
        return full.panel(ALL, START, f.index[-1]).ret

    cuts = sessions[[60, 130, 200, 260, 330, 400]]  # spans the 2016-03-15 split and three dividends
    report = truncation_invariance(ret_through_pipeline, frame, cuts, atol=1e-12)
    assert report.passed, report.violations


def test_tr_close_anchored_at_a_date_ignores_later_history(full):
    a = full.panel(["BBB"], START, "2016-12-30").tr_close(anchor="2015-12-31")
    b = full.panel(["BBB"], START, "2015-12-31").tr_close()
    pd.testing.assert_frame_equal(a, b)


def test_missing_tickers_are_reported_in_dropped(full):
    p = full.panel(["AAA", "NOPE"], START, "2016-12-30")
    assert p.dropped == {"NOPE": "not in store"} and list(p.close.columns) == ["AAA"]
    early = full.panel(ALL, START, "2015-03-02")  # CCC lists in September: still a (NaN) column
    assert list(early.close.columns) == ALL and early.empty == ["CCC"]
    assert early.close["CCC"].isna().all()


def test_late_listing_and_delisting_appear_as_nan_and_are_ineligible(full):
    p = full.panel(ALL, START, "2016-12-30")
    assert (
        p.close["CCC"].loc[:"2015-08-31"].isna().all()
        and p.close["CCC"].loc["2015-09-01":].notna().all()
    )
    assert p.close["DDD"].loc["2016-02-29":].isna().all()
    assert p.eligible(START, "2016-12-30", min_coverage=0.95) == ["AAA", "BBB"]
    assert p.eligible("2015-10-01", "2016-02-26", min_coverage=0.95) == ["AAA", "BBB", "CCC", "DDD"]


def test_members_mask_follows_point_in_time_membership(tmp_path, calendar):
    f = tmp_path / "m.csv"
    f.write_text("ticker,start,end\nAAA,,2015-12-31\nBBB,,\nCCC,2016-01-04,\n")
    cfg = make_cfg(tmp_path).model_copy(
        update={"universe": UniverseConfig(mode="membership_file", membership_file=str(f))}
    )
    pipe = DataPipeline(cfg, source=SyntheticSource(make_world()), calendar=calendar)
    try:
        pipe.refresh(["AAA", "BBB", "CCC"])
        p = pipe.panel(["AAA", "BBB", "CCC"], "2015-12-29", "2016-01-06")
        mask = pipe.members_mask(p)
        # AAA's interval is half-open: removed *effective* 2015-12-31, so out on that date
        assert mask.loc["2015-12-30"].to_dict() == {"AAA": True, "BBB": True, "CCC": False}
        assert mask.loc["2015-12-31"].to_dict() == {"AAA": False, "BBB": True, "CCC": False}
        assert mask.loc["2016-01-04"].to_dict() == {"AAA": False, "BBB": True, "CCC": True}
        assert list(mask.columns) == list(p.close.columns)
    finally:
        pipe.close()


def test_audit_of_clean_synthetic_data_has_no_warnings_or_errors(full):
    issues = full.audit()
    assert [i for i in issues if i.severity != Severity.INFO] == []


def test_audit_catches_a_corrupted_store(full):
    prices = full.store.read_prices("BBB")
    prices = prices.drop(prices.index[100:110])  # ten-session hole
    full.store.write_history("BBB", prices, full.store.read_actions("BBB"))
    issues = [i for i in full.audit(["BBB"]) if i.code == "MISSING_SESSIONS"]
    assert len(issues) == 1 and issues[0].severity == Severity.ERROR and issues[0].count == 10


def test_pipeline_is_deterministic_and_the_fingerprint_tracks_the_data(make_pipeline):
    a, b = make_pipeline(subdir="a"), make_pipeline(subdir="b")
    a.refresh(), b.refresh()
    assert a.fingerprint() == b.fingerprint() and len(a.fingerprint()) == 16
    other = make_pipeline(subdir="c", world=make_world(seed=8))
    other.refresh()
    assert other.fingerprint() != a.fingerprint()
    assert a.fingerprint(["AAA"]) != a.fingerprint(["BBB"]) and a.fingerprint(
        ["AAA"]
    ) == b.fingerprint(["AAA"])


def test_dataset_manifest_records_what_a_result_was_computed_on(full):
    path = full.write_dataset_manifest()
    m = json.loads(path.read_text())
    assert m["fingerprint"] == full.fingerprint() and m["n_tickers"] == 4
    assert m["first_date"] == "2015-01-02" and m["last_date"] == "2016-12-30"
    assert m["config"]["universe"]["mode"] == "static"
    assert any("survivorship" in c.lower() for c in m["universe"]["caveats"])
    assert {"pandas", "numpy", "pyarrow"} <= set(m["versions"]) and "git_commit" in m


def test_universe_is_built_once_and_can_be_rebuilt(full):
    assert full.universe() is full.universe()
    assert full.universe(rebuild=True) is not None


def test_raw_close_reproduces_what_actually_printed(make_pipeline, full):
    """As-traded prices from a post-split store equal the closes a pre-split vendor delivered."""
    before = make_pipeline(vendor_date="2016-03-14", subdir="pre_split")
    before.refresh()
    printed = before.store.read_prices("AAA")["close"]  # the vendor's numbers on 2016-03-14
    got = full.raw_close("AAA", START, "2016-03-14")
    pd.testing.assert_series_equal(got, printed, check_freq=False)
    assert full.raw_close("AAA", "2016-03-15", "2016-03-15").iloc[0] == pytest.approx(
        full.store.read_prices("AAA")["close"].loc["2016-03-15"]
    )  # the ex-date and later are already on the post-split price
    with pytest.raises(KeyError):
        full.raw_close("NOPE", START, "2016-03-14")


def test_verified_members_mask_drops_intervals_whose_prices_are_another_companys(make_pipeline):
    pipe = make_pipeline(subdir="ident")
    pipe.refresh(["AAA", "BBB"])
    stub = pipe.store.read_prices("BBB").assign(volume=0.0)  # a reused symbol: never trades
    pipe.store.write_history("BBB", stub, pipe.store.read_actions("BBB"))
    p = pipe.panel(["AAA", "BBB"], "2015-06-01", "2015-06-05")
    assert pipe.members_mask(p).all().all()  # membership alone says both are in
    verified = pipe.members_mask(p, verified=True)
    assert verified["AAA"].all() and not verified["BBB"].any()
    assert pipe.identity_report().set_index("ticker").loc["BBB", "verdict"] == "suspect"
