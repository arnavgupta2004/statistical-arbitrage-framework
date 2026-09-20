from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from statarb.config import DataConfig, RefreshConfig, UniverseConfig
from statarb.data.cleaning.calendar import TradingCalendar
from statarb.data.pipeline import DataPipeline
from statarb.data.sources.synthetic import SyntheticSource, SyntheticWorld

WORLD_START = "2015-01-02"
WORLD_END = "2016-12-30"
SPLIT_DATE = "2016-03-15"  # AAA 2-for-1
DIVIDEND_DATES = ["2015-06-15", "2016-01-14", "2016-09-14"]


@pytest.fixture(scope="session")
def calendar() -> TradingCalendar:
    return TradingCalendar("XNYS")


def make_world(**overrides) -> SyntheticWorld:
    kw = dict(
        tickers=["AAA", "BBB", "CCC", "DDD"],
        start=WORLD_START,
        end=WORLD_END,
        seed=7,
        splits={"AAA": [(SPLIT_DATE, 2.0)]},
        dividends={
            "AAA": [(DIVIDEND_DATES[0], 0.5), (DIVIDEND_DATES[2], 0.6)],
            "BBB": [(d, 0.4) for d in DIVIDEND_DATES],
        },
        lifespans={"CCC": ("2015-09-01", None), "DDD": (None, "2016-02-26")},
    )
    kw.update(overrides)
    return SyntheticWorld(**kw)


@pytest.fixture
def world() -> SyntheticWorld:
    return make_world()


def make_cfg(tmp_path, tickers=("AAA", "BBB", "CCC", "DDD"), **refresh) -> DataConfig:
    return DataConfig(
        store_dir=str(tmp_path / "store"),
        source="synthetic",
        start=date(2015, 1, 2),
        refresh=RefreshConfig(request_pause_s=0.0, **refresh),
        universe=UniverseConfig(mode="static", tickers=list(tickers)),
    )


@pytest.fixture
def cfg(tmp_path) -> DataConfig:
    return make_cfg(tmp_path)


@pytest.fixture
def make_pipeline(tmp_path, calendar):
    """Factory: ``make_pipeline(vendor_date, subdir)`` -> DataPipeline over a fresh store."""
    made: list[DataPipeline] = []

    def _make(vendor_date=None, subdir="store", world: SyntheticWorld | None = None, **refresh):
        cfg = make_cfg(tmp_path / subdir, **refresh)
        source = SyntheticSource(world or make_world(), vendor_date=vendor_date)
        pipe = DataPipeline(cfg, source=source, calendar=calendar)
        made.append(pipe)
        return pipe

    yield _make
    for p in made:
        p.close()


def ts(x) -> pd.Timestamp:
    return pd.Timestamp(x)
