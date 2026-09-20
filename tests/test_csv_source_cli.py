"""CSV source (manual Stooq-style files) and the command line, driven end to end."""

from __future__ import annotations

import pytest
import yaml

from statarb.data.cli import main
from statarb.data.sources.base import SourceError
from statarb.data.sources.csv_source import CsvDirectorySource
from statarb.data.sources.synthetic import SyntheticSource

from .conftest import make_world, ts


@pytest.fixture
def csv_dir(tmp_path):
    """Write a Stooq-format export of the synthetic world (lower-case ``.us`` names)."""
    d = tmp_path / "csv"
    d.mkdir()
    src = SyntheticSource(make_world())
    for t in ("AAA", "BBB"):
        raw = src.fetch(t, ts("2015-01-01"), ts("2016-12-30"))
        out = raw.prices.drop(columns="adj_close").reset_index()
        out.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        out.to_csv(d / f"{t.lower()}.us.csv", index=False)
        raw.actions.to_csv(d / f"{t.lower()}.us.actions.csv", index=False)
    return d


def test_csv_source_reads_prices_actions_and_filters_the_range(csv_dir):
    s = CsvDirectorySource(csv_dir, suffix=".us")
    h = s.fetch("AAA", ts("2016-01-04"), ts("2016-06-30"))
    assert h.prices.index.min() == ts("2016-01-04") and h.prices.index.max() == ts("2016-06-30")
    assert list(h.prices.columns) == ["open", "high", "low", "close", "volume"]
    assert h.actions["kind"].tolist() == ["split"] and h.actions["value"].tolist() == [2.0]


def test_csv_source_unknown_ticker_is_empty_not_an_error(csv_dir):
    assert (
        CsvDirectorySource(csv_dir, suffix=".us")
        .fetch("ZZZ", ts("2015-01-01"), ts("2016-01-01"))
        .prices.empty
    )


def test_csv_source_rejects_malformed_files(tmp_path):
    (tmp_path / "bad.csv").write_text("foo,bar\n1,2\n")
    with pytest.raises(SourceError, match="lacks columns"):
        CsvDirectorySource(tmp_path).fetch("bad", ts("2020-01-01"), ts("2020-02-01"))


@pytest.fixture
def cli_config(tmp_path, csv_dir):
    cfg = {
        "store_dir": str(tmp_path / "store"),
        "source": "csv",
        "source_options": {"directory": str(csv_dir), "suffix": ".us"},
        "start": "2015-01-02",
        "universe": {"mode": "static", "tickers": ["AAA", "BBB", "ZZZ"]},
        "refresh": {"request_pause_s": 0.0},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_cli_end_to_end(cli_config, capsys):
    assert main(["universe", "--config", cli_config]) == 0
    out = capsys.readouterr().out
    assert "static" in out and "CAVEAT" in out and "STATIC LIST" in out

    assert main(["refresh", "--config", cli_config]) == 0
    out = capsys.readouterr().out
    assert "'full': 2" in out and "'no_data': 1" in out  # ZZZ has no file

    # the export ends in 2016 but "today" is later: the first re-run probes once for newer data ...
    assert main(["refresh", "--config", cli_config]) == 0
    assert "'incremental': 2, 'skipped': 1" in capsys.readouterr().out
    # ... finds none, and thereafter backs off instead of re-asking about a history that has ended
    assert main(["refresh", "--config", cli_config]) == 0
    assert "'skipped': 3" in capsys.readouterr().out

    assert main(["status", "--config", cli_config]) == 0
    out = capsys.readouterr().out
    assert "ok" in out and "no_data" in out and "fingerprint:" in out

    assert main(["audit", "--config", cli_config]) == 0
    assert "findings written" in capsys.readouterr().out

    assert main(["snapshot", "--config", cli_config]) == 0
    assert "fingerprint:" in capsys.readouterr().out

    assert main(["survivorship", "--config", cli_config]) == 0
    out = capsys.readouterr().out
    assert "reliable_from" in out and "CAVEAT" in out


def test_config_rejects_unknown_keys(tmp_path):
    from statarb.config import load_config

    p = tmp_path / "bad.yaml"
    p.write_text("refresh:\n  overlap_dayz: 5\n")
    with pytest.raises(Exception, match="overlap_dayz"):
        load_config(p)


def test_default_config_loads():
    from statarb.config import load_config

    cfg = load_config("configs/data_default.yaml")
    assert (
        cfg.source == "yahoo" and cfg.universe.mode == "sp500_wikipedia" and cfg.calendar == "XNYS"
    )
