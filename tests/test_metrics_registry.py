from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from statarb.backtest import metrics as m
from statarb.research.registry import Registry, spec_id


# ---- metrics: hand-worked values ---------------------------------------------------------------
def test_sharpe_sortino_by_hand():
    r = np.array([0.01, -0.02, 0.03, 0.0, -0.01])
    assert m.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert m.sortino(r) == pytest.approx(r.mean() / downside * np.sqrt(252))
    assert np.isnan(m.sharpe(np.zeros(5))) and np.isnan(m.sortino(np.array([0.01, 0.02])))


def test_drawdown_by_hand():
    r = np.array([0.10, -0.20, 0.05, 0.05, 0.30])
    eq = np.cumprod(1 + r)  # 1.10, 0.88, 0.924, 0.9702, 1.26126
    dd = m.drawdowns(r)
    assert dd == pytest.approx([0.0, 0.88 / 1.10 - 1, 0.924 / 1.10 - 1, 0.9702 / 1.10 - 1, 0.0])
    assert m.max_drawdown(r) == pytest.approx(0.88 / 1.10 - 1)
    assert m.drawdown_duration(r) == 3  # bars 1-3 below the 1.10 peak
    assert m.max_drawdown(np.array([0.01, 0.02])) == 0.0 and eq[-1] == pytest.approx(1.26126)
    # a loss on the very first day is a drawdown from the STARTING equity (1.0), not from the first close
    assert m.max_drawdown(np.array([-0.10, 0.05])) == pytest.approx(-0.10)
    assert m.drawdowns(np.array([-0.10, 0.05])) == pytest.approx([-0.10, 0.945 - 1])


def test_calmar_profit_factor_hit_rate():
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.0, 0.01])
    assert m.profit_factor(r) == pytest.approx(0.06 / 0.03)
    assert m.hit_rate(r) == pytest.approx(3 / 5)  # the flat day is not a miss
    assert m.calmar(r) == pytest.approx(r.mean() * 252 / abs(m.max_drawdown(r)))
    assert m.profit_factor(np.array([0.01, 0.02])) == float("inf") and np.isnan(
        m.hit_rate(np.zeros(3))
    )
    assert np.isnan(m.calmar(np.array([0.01, 0.02])))


def test_summary_and_by_year():
    idx = pd.bdate_range("2019-06-01", "2020-12-31")
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, len(idx)), index=idx)
    s = m.performance_summary(r.to_numpy(), turnover=np.full(len(r), 0.1))
    assert s["n_days"] == len(r) and s["turnover_per_year"] == pytest.approx(0.1 * 252, rel=0.01)
    for k in (
        "sharpe",
        "sortino",
        "max_drawdown_pct",
        "drawdown_duration_days",
        "calmar",
        "hit_rate",
        "profit_factor",
    ):
        assert k in s
    yr = m.by_year(r)
    assert yr["year"].tolist() == [2019, 2020] and yr["n_days"].sum() == len(r)


# ---- registry ---------------------------------------------------------------------------------------
def rec(**kw):
    base = dict(
        stage=5,
        kind="strategy",
        name="wf",
        phases=["research"],
        windows={"test": "2015"},
        parameters={"entry": 2.0},
        seed=1,
    )
    base.update(kw)
    return base


def test_the_registry_is_append_only_json_lines_with_all_required_fields(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    a = reg.register(**rec(results={"sharpe": 0.4}, data_fingerprint="abc123", strategy="pairs"))
    b = reg.register(**rec(name="wf2"))
    lines = (tmp_path / "r.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["spec_id"] == a["spec_id"] != b["spec_id"]
    for k in (
        "date",
        "stage",
        "kind",
        "name",
        "universe",
        "phases",
        "windows",
        "strategy",
        "parameters",
        "features",
        "costs",
        "results",
        "seed",
        "git_commit",
        "data_fingerprint",
        "spec_id",
        "run",
    ):
        assert k in a, k
    assert a["costs"] == {"model": "none (gross)"} and a["phases"] == ["research"]
    assert [r["name"] for r in reg.records()] == ["wf", "wf2"]


def test_spec_id_identifies_what_was_run_not_what_it_returned():
    a, b = rec(results={"sharpe": 1.0}), rec(results={"sharpe": -3.0}, notes="x")
    assert spec_id(a) == spec_id(b)  # results, notes, dates and commit do not change the spec
    assert spec_id(rec(parameters={"entry": 2.5})) != spec_id(a) and spec_id(
        rec(seed=2)
    ) != spec_id(a)


def test_reruns_are_recorded_not_overwritten_and_counting_distinguishes_them(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    first = reg.register(**rec())
    again = reg.register(**rec(results={"sharpe": 9}))
    reg.register(**rec(parameters={"entry": 2.5}))
    reg.register(**rec(kind="simulation", name="sim"))
    assert (first["run"], again["run"]) == (1, 2)
    assert reg.count_trials("strategy") == {"distinct_specs": 2, "total_runs": 3}
    assert reg.count_trials("simulation") == {"distinct_specs": 1, "total_runs": 1}
    assert reg.count_trials("strategy", phase="validation") == {
        "distinct_specs": 0,
        "total_runs": 0,
    }
    reg.register(**rec(name="v", phases=["research", "validation"]))
    assert reg.count_trials("strategy", phase="validation")["distinct_specs"] == 1


def test_registry_validation_and_an_empty_registry(tmp_path):
    reg = Registry(tmp_path / "none.jsonl")
    assert reg.records() == [] and reg.count_trials() == {"distinct_specs": 0, "total_runs": 0}
    with pytest.raises(ValueError, match="kind"):
        reg.register(**rec(kind="guess"))
    with pytest.raises(ValueError, match="phases"):
        reg.register(**rec(phases=["training"]))


def test_rolling_sharpe_and_volatility_are_causal_and_match_a_direct_computation():
    from statarb.backtest.metrics import rolling_sharpe, rolling_volatility, sharpe

    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.01, 400), index=pd.bdate_range("2020-01-01", periods=400))
    rs, rv = rolling_sharpe(r, 100), rolling_volatility(r, 50)
    assert rs.iloc[:99].isna().all() and rv.iloc[:49].isna().all()
    for t in (99, 250, 399):
        assert rs.iloc[t] == pytest.approx(sharpe(r.iloc[t - 99 : t + 1].to_numpy()))
        assert rv.iloc[t] == pytest.approx(r.iloc[t - 49 : t + 1].std(ddof=1) * np.sqrt(252))
    p = r.copy()
    p.iloc[300:] *= 5.0  # the future changes...
    assert rolling_sharpe(p, 100).iloc[:300].equals(rs.iloc[:300])  # ...the past does not
    flat = rolling_sharpe(pd.Series(np.zeros(200)), 50)
    assert flat.isna().all()  # a flat window has no Sharpe
