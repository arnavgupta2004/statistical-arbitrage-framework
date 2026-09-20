"""Symbol-reuse detection: is the stored series plausibly the index member's own?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.data.universe.builders import Universe
from statarb.data.universe.identity import check_identity
from statarb.data.universe.membership import Membership
from statarb.data.universe.survivorship import coverage

from .conftest import ts

SESSIONS = pd.bdate_range("2020-01-01", "2020-12-31")


def series(dates, volume=1000.0) -> pd.DataFrame:
    n = len(dates)
    c = np.linspace(100, 110, n)
    return pd.DataFrame(
        {"open": c, "high": c, "low": c, "close": c, "volume": np.full(n, volume)},
        index=pd.DatetimeIndex(dates, name="date"),
    )


def universe(rows) -> Universe:
    return Universe(
        Membership(pd.DataFrame(rows, columns=["ticker", "start", "end"])), pd.Series(dtype=object)
    )


@pytest.fixture
def store_data():
    stub = series(SESSIONS, volume=0.0)  # a reused symbol: stale, never traded
    stub.iloc[::20, stub.columns.get_loc("volume")] = 5  # a few prints
    return {
        "GOOD": series(SESSIONS),
        "STUB": stub,
        "DELISTED": series(SESSIONS[:120]),  # real data that simply ends early
    }


def run(uni, data):
    return check_identity(uni, data.get, SESSIONS, "2020-01-01", "2020-12-31").set_index("ticker")


def test_verdicts(store_data):
    uni = universe(
        [
            ("GOOD", pd.NaT, pd.NaT),
            ("STUB", pd.NaT, pd.NaT),
            ("DELISTED", pd.NaT, pd.NaT),
            ("GONE", pd.NaT, pd.NaT),
        ]
    )
    r = run(uni, store_data)
    assert r.loc["GOOD", "verdict"] == "ok" and r.loc["GOOD", "obs_share"] == 1.0
    assert r.loc["STUB", "verdict"] == "suspect" and r.loc["STUB", "volume_share"] < 0.1
    assert r.loc["GONE", "verdict"] == "no_prices" and r.loc["GONE", "n_obs"] == 0
    # data that ends early is *partial*, not suspect: obs_share tells you, the verdict does not condemn it
    assert r.loc["DELISTED", "verdict"] == "ok" and 0.4 < r.loc["DELISTED", "obs_share"] < 0.6


def test_only_the_membership_interval_is_judged(store_data):
    """A symbol that was junk *before* the company joined the index is irrelevant to that interval."""
    mixed = pd.concat([series(SESSIONS[:100], volume=0.0), series(SESSIONS[100:])])
    uni = universe([("MIXED", SESSIONS[100], pd.NaT)])
    assert run(uni, {"MIXED": mixed}).loc["MIXED", "verdict"] == "ok"
    uni_early = universe([("MIXED", pd.NaT, SESSIONS[100])])
    assert run(uni_early, {"MIXED": mixed}).loc["MIXED", "verdict"] == "suspect"


def test_end_of_interval_is_exclusive():
    """Removed effective D: the bar on D is outside the interval and must not be judged."""
    data = pd.concat([series(SESSIONS[:50]), series(SESSIONS[50:], volume=0.0)])
    uni = universe([("X", pd.NaT, SESSIONS[50])])
    r = run(uni, {"X": data}).loc["X"]
    assert r["verdict"] == "ok" and r["n_obs"] == 50


def test_coverage_excludes_suspect_and_missing_members(store_data):
    uni = universe([("GOOD", pd.NaT, pd.NaT), ("STUB", pd.NaT, pd.NaT), ("GONE", pd.NaT, pd.NaT)])
    manifest = pd.DataFrame(
        {
            "ticker": ["GOOD", "STUB"],
            "status": ["ok", "ok"],
            "first_date": [SESSIONS[0], SESSIONS[0]],
            "last_date": [SESSIONS[-1], SESSIONS[-1]],
        }
    )
    d = [ts("2020-06-01")]
    naive = coverage(uni, manifest, d)
    aware = coverage(uni, manifest, d, identity=run(uni, store_data).reset_index())
    assert naive.iloc[0]["n_with_prices"] == 2  # counts the stub as "we have prices"
    assert aware.iloc[0]["n_with_prices"] == 1 and aware.iloc[0]["coverage"] == pytest.approx(1 / 3)
