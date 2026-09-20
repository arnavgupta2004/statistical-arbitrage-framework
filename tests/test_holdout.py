"""The sealed holdout: guard on every research read, one-time unlock, auditable ledger."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from statarb.config import DataConfig, SplitConfig, load_config
from statarb.data.holdout import HoldoutAlreadySpent, HoldoutLedger, HoldoutViolation
from statarb.data.pipeline import DataPipeline
from statarb.data.sources.synthetic import SyntheticSource, SyntheticWorld

from .conftest import make_cfg, make_world

SPLIT = SplitConfig(
    data_start=date(2015, 1, 2),
    research_end=date(2015, 12, 31),
    validation_start=date(2016, 1, 1),
    validation_end=date(2016, 6, 30),
    holdout_start=date(2016, 7, 1),
)


@pytest.fixture
def guarded(tmp_path, calendar):
    cfg = make_cfg(tmp_path).model_copy(
        update={"split": SPLIT.model_copy(update={"ledger_path": str(tmp_path / "ledger.jsonl")})}
    )
    pipe = DataPipeline(cfg, source=SyntheticSource(make_world()), calendar=calendar)
    pipe.refresh()
    yield pipe
    pipe.close()


def test_split_config_validates_ordering_and_assigns_phases():
    assert (
        SPLIT.phase_of("2015-06-01") == "research" and SPLIT.phase_of("2016-01-01") == "validation"
    )
    assert (
        SPLIT.phase_of("2016-06-30") == "validation" and SPLIT.phase_of("2016-07-01") == "holdout"
    )
    with pytest.raises(ValidationError, match="validation_start"):
        SplitConfig(
            data_start=date(2015, 1, 1),
            research_end=date(2015, 12, 31),
            validation_start=date(2015, 6, 1),
            validation_end=date(2016, 6, 30),
            holdout_start=date(2016, 7, 1),
        )
    with pytest.raises(ValidationError):
        SplitConfig(
            data_start=date(2015, 1, 1),
            research_end=date(2015, 12, 31),
            validation_start=date(2016, 1, 1),
            validation_end=date(2016, 6, 30),
            holdout_start=date(2016, 6, 1),
        )


def test_the_default_config_carries_the_documented_split():
    cfg = load_config("configs/data_default.yaml")
    s = cfg.split
    assert (
        s.data_start,
        s.research_end,
        s.validation_start,
        s.validation_end,
        s.holdout_start,
    ) == (
        date(2011, 1, 3),
        date(2018, 12, 31),
        date(2019, 1, 1),
        date(2021, 12, 31),
        date(2022, 1, 1),
    )


def test_reads_before_the_holdout_are_allowed_and_reads_into_it_are_refused(guarded):
    p = guarded.panel(["AAA"], "2015-01-02", "2016-06-30")  # last research/validation day
    assert p.close.index[-1] <= p.close.index.max()
    with pytest.raises(HoldoutViolation, match="sealed holdout"):
        guarded.panel(["AAA"], "2015-01-02", "2016-07-01")
    with pytest.raises(HoldoutViolation):
        guarded.panel(["AAA"], "2015-01-02", "2016-12-30")


def test_as_of_cannot_smuggle_holdout_information_in(guarded):
    """A later as_of would rebase prices with splits that happen inside the holdout (AAA splits 2016-03-15
    in the synthetic world; here as_of past the holdout start is refused outright)."""
    with pytest.raises(HoldoutViolation):
        guarded.panel(["AAA"], "2015-01-02", "2015-12-31", as_of="2016-08-01")
    guarded.panel(["AAA"], "2015-01-02", "2015-12-31", as_of="2016-06-30")  # before the seal: fine


def test_the_other_research_reads_are_guarded_too(guarded):
    with pytest.raises(HoldoutViolation):
        guarded.raw_close("AAA", "2015-01-02", "2016-12-30")
    with pytest.raises(HoldoutViolation):
        guarded.identity_report(as_of="2016-12-30")
    with pytest.raises(HoldoutViolation):
        guarded.identity_report()  # default = today, which is inside the holdout
    guarded.identity_report(as_of="2016-06-30")


def test_without_a_split_nothing_is_guarded(make_pipeline):
    pipe = make_pipeline()
    pipe.refresh()
    assert pipe.cfg.split is None and pipe.ledger is None
    pipe.panel(["AAA"], "2015-01-02", "2016-12-30")
    with pytest.raises(ValueError, match="no split"):
        pipe.unlock_holdout("x")


def test_unlocking_is_one_time_recorded_and_audited(guarded, tmp_path):
    ledger = HoldoutLedger(tmp_path / "ledger.jsonl")
    assert ledger.events() == [] and not ledger.is_spent()
    with pytest.raises(ValueError, match="purpose"):
        guarded.unlock_holdout("  ")
    event = guarded.unlock_holdout("final evaluation of the frozen strategy")
    assert event["event"] == "unlock" and "final evaluation" in event["purpose"]
    assert (
        guarded.panel(["AAA"], "2015-01-02", "2016-12-30").close.index[-1].year == 2016
    )  # now readable
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["data_fingerprint"] == guarded.fingerprint()
    assert ledger.is_spent()


def test_a_second_unlock_is_refused_unless_recorded_as_a_contaminating_reopen(guarded, tmp_path):
    guarded.unlock_holdout("first and only look")
    with pytest.raises(HoldoutAlreadySpent):
        guarded.unlock_holdout("another look")
    ev = guarded.unlock_holdout("re-run after a bug", allow_reopen=True)
    assert ev["event"] == "reopen"
    assert [e["event"] for e in HoldoutLedger(tmp_path / "ledger.jsonl").events()] == [
        "unlock",
        "reopen",
    ]


def test_a_new_pipeline_on_a_spent_ledger_is_still_sealed_until_it_unlocks(
    guarded, tmp_path, calendar
):
    guarded.unlock_holdout("the one look")
    cfg = guarded.cfg
    other = DataPipeline(cfg, source=SyntheticSource(SyntheticWorld(tickers=[])), calendar=calendar)
    try:
        with pytest.raises(
            HoldoutViolation
        ):  # another process starts sealed: unlocking is per instance
            other.panel(["AAA"], "2015-01-02", "2016-12-30")
        with pytest.raises(HoldoutAlreadySpent):
            other.unlock_holdout("a different session")
    finally:
        other.close()


def test_data_config_accepts_a_missing_split():
    assert DataConfig().split is None
