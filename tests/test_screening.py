"""Pair screen: null construction, calibration, selection rule, eligibility and causality."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from statarb.config import DataConfig, RefreshConfig, UniverseConfig
from statarb.data.pipeline import DataPipeline
from statarb.data.sources.synthetic import SyntheticSource, SyntheticWorld
from statarb.models.ou import simulate_ou
from statarb.selection.screening import (
    ScreenConfig,
    ScreenResult,
    discover_pairs,
    empirical_pvalue,
    screen_pairs,
    shift_panel,
)

# ---- synthetic panels -----------------------------------------------------------------------


def factor_panel(n=500, n_stocks=36, seed=0, planted=0, sector_size=12, fast=0):
    """Log prices with a market factor and idiosyncratic random walks (no cointegration), plus
    ``planted`` cointegrated (Y, X) pairs: Y = 0.3 + 0.7 X + OU(kappa=0.08)."""
    rng = np.random.default_rng(seed)
    mkt = np.cumsum(rng.normal(0, 0.01, n))
    cols, sectors = {}, {}
    for i in range(n_stocks):
        name = f"S{i:02d}"
        cols[name] = 4 + rng.uniform(0.5, 1.2) * mkt + np.cumsum(rng.normal(0, 0.015, n))
        sectors[name] = f"sec{i // sector_size}"
    for k in range(planted):
        x = 4 + 0.8 * mkt + np.cumsum(rng.normal(0, 0.02, n))
        cols[f"X{k}"], cols[f"Y{k}"] = x, 0.3 + 0.7 * x + simulate_ou(0.08, 0, 0.02, n, rng=rng)
        sectors[f"X{k}"] = sectors[f"Y{k}"] = "sec0"
    for k in range(fast):  # cointegrated but reverting in < 1 day: significant, yet untradeable
        x = 4 + 0.8 * mkt + np.cumsum(rng.normal(0, 0.02, n))
        cols[f"XF{k}"], cols[f"YF{k}"] = x, 0.3 + 0.7 * x + simulate_ou(1.2, 0, 0.02, n, rng=rng)
        sectors[f"XF{k}"] = sectors[f"YF{k}"] = "sec0"
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(cols, index=idx), pd.Series(sectors)


FAST = dict(n_null_panels=6, k_neighbours=4, seed=3)

# ---- null construction ----------------------------------------------------------------------


def test_shift_panel_preserves_every_stocks_return_distribution_and_start_level():
    logp, _ = factor_panel()
    shifted = shift_panel(logp, np.random.default_rng(1))
    for c in logp.columns:
        assert np.allclose(np.sort(logp[c].diff().dropna()), np.sort(shifted[c].diff().dropna()))
    assert np.allclose(shifted.iloc[0], logp.iloc[0]) and not shifted.isna().any().any()
    assert shifted.index.equals(logp.index) and list(shifted.columns) == list(logp.columns)


def test_shift_panel_destroys_cross_sectional_dependence():
    logp, _ = factor_panel(n=800)
    upper = np.triu_indices(logp.shape[1], 1)

    def mean_abs_corr(p):
        return np.abs(np.corrcoef(p.diff().dropna().to_numpy(), rowvar=False)[upper]).mean()

    assert mean_abs_corr(logp) > 0.15  # market factor: clearly dependent (measured ~0.22)
    assert mean_abs_corr(shift_panel(logp, np.random.default_rng(2))) < 0.06  # ~ sampling noise


def test_shift_panel_is_deterministic_and_uses_different_shifts_per_stock():
    logp, _ = factor_panel()
    a = shift_panel(logp, np.random.default_rng(5))
    b = shift_panel(logp, np.random.default_rng(5))
    c = shift_panel(logp, np.random.default_rng(6))
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.to_numpy(), c.to_numpy())
    r0, r1 = logp.diff().dropna(), a.diff().dropna()
    same_alignment = [np.allclose(r0[col], r1[col]) for col in logp.columns]
    assert sum(same_alignment) <= 1  # essentially every column really moved


def test_empirical_pvalue_formula_bounds_and_monotonicity():
    null = np.array([-5.0, -3.0, -1.0, 0.0])
    p = empirical_pvalue([-6.0, -4.0, -3.0, 9.0], null)
    assert p == pytest.approx([1 / 5, 2 / 5, 3 / 5, 1.0])  # (1 + #null <= t) / (1 + B)
    assert empirical_pvalue(-100.0, null) == pytest.approx(
        1 / 5
    )  # the floor is 1/(1+B), never zero
    grid = np.linspace(-8, 2, 50)
    assert (np.diff(empirical_pvalue(grid, null)) >= 0).all()


# ---- the screen -----------------------------------------------------------------------------


def test_screen_finds_planted_pairs_and_ranks_them_by_significance():
    logp, sectors = factor_panel(planted=3, seed=1)
    res = screen_pairs(logp, sectors, ScreenConfig(**FAST))
    tested = res.pairs.set_index(["y", "x"])
    found = 0
    for k in range(3):
        row = tested[
            (tested.index.get_level_values("y").isin([f"Y{k}", f"X{k}"]))
            & tested.index.get_level_values("x").isin([f"Y{k}", f"X{k}"])
        ]
        assert len(row) == 1
        found += bool(row["p_calibrated"].iloc[0] <= 0.05)
    assert found == 3
    ranked = res.pairs.reset_index(drop=True)
    for k in range(3):  # a planted pair sits near the top; a chance pair may outrank a weak one
        i = ranked.index[ranked[["y", "x"]].isin([f"Y{k}", f"X{k}"]).all(axis=1)][0]
        assert i < 10 and ranked.at[i, "T"] < -3.5 and ranked.at[i, "beta"] > 0


def test_recovers_the_hedge_ratio_and_half_life_of_a_planted_pair():
    """One draw has ~1 day of half-life noise (measured sd 1.1 over 40 seeds), so average seeds."""
    betas, half_lives = [], []
    for seed in range(6):
        logp, sectors = factor_panel(n=1500, planted=1, seed=20 + seed, n_stocks=10)
        res = screen_pairs(logp, sectors, ScreenConfig(n_null_panels=1, k_neighbours=4, seed=1))
        row = res.pairs[res.pairs[["y", "x"]].isin(["Y0", "X0"]).all(axis=1)].iloc[0]
        betas.append(
            row["beta"] if row["y"] == "Y0" else 1 / row["beta"]
        )  # express as Y-on-X slope
        half_lives.append(row["half_life"])
    assert np.mean(betas) == pytest.approx(0.7, rel=0.05)
    assert np.mean(half_lives) == pytest.approx(np.log(2) / 0.08, rel=0.2)  # 8.7 days


def test_selection_rule_is_applied_exactly_and_ranked_by_the_economic_proxy():
    logp, sectors = factor_panel(planted=3, seed=1, fast=1)
    cfg = ScreenConfig(**FAST, half_life_min=5, half_life_max=60, max_pairs=20)
    res = screen_pairs(logp, sectors, cfg)
    sel = res.selected
    assert len(sel) >= 3
    assert (sel["p_calibrated"] <= cfg.alpha).all() and (sel["beta"] > 0).all()
    assert sel["half_life"].between(cfg.half_life_min, cfg.half_life_max).all()
    assert sel["rank"].tolist() == list(range(1, len(sel) + 1))
    assert sel["edge_bps_per_day"].is_monotonic_decreasing
    # every non-selected row fails at least one criterion
    rest = res.pairs[~res.pairs["selected"]]
    ok = (
        (rest["p_calibrated"] <= cfg.alpha)
        & rest["half_life"].between(cfg.half_life_min, cfg.half_life_max)
        & (rest["beta"] > 0)
    )
    assert not ok.any()
    fast = res.pairs[res.pairs[["y", "x"]].isin(["YF0", "XF0"]).all(axis=1)].iloc[0]
    assert (
        fast["p_calibrated"] <= cfg.alpha and fast["half_life"] < cfg.half_life_min
    )  # ...and rejected
    assert not fast["selected"]
    one = screen_pairs(logp, sectors, ScreenConfig(**FAST, max_pairs=1)).selected
    assert len(one) == 1 and one["edge_bps_per_day"].iloc[0] == pytest.approx(
        sel["edge_bps_per_day"].iloc[0]
    )


def test_the_screen_is_deterministic_for_a_seed_and_seed_changes_the_null_only():
    logp, sectors = factor_panel(planted=2, seed=2)
    a = screen_pairs(logp, sectors, ScreenConfig(**FAST)).pairs
    b = screen_pairs(logp, sectors, ScreenConfig(**FAST)).pairs
    pd.testing.assert_frame_equal(a, b)
    c = screen_pairs(logp, sectors, ScreenConfig(**{**FAST, "seed": 99})).pairs
    assert (
        a["T"].sort_values().tolist() == c["T"].sort_values().tolist()
    )  # real statistics unchanged
    assert not np.allclose(
        a.sort_values(["y", "x"])["p_calibrated"], c.sort_values(["y", "x"])["p_calibrated"]
    )


def test_calibration_holds_on_a_factor_null_panel_and_the_naive_pvalue_does_not():
    """No cointegration exists here (common market factor + idiosyncratic random walks), but the
    candidates are chosen for high correlation and the better of two directions is kept.  The
    calibrated p-value must reject ~5 %; the naive best-of-two MacKinnon p-value must reject ~10 %."""
    cal, naive, n = 0, 0, 0
    for seed in range(6):
        logp, sectors = factor_panel(n=500, n_stocks=36, seed=100 + seed)
        res = screen_pairs(logp, sectors, ScreenConfig(n_null_panels=8, k_neighbours=4, seed=seed))
        cal += (res.pairs["p_calibrated"] <= 0.05).sum()
        naive += (res.pairs["p_nominal_best"] < 0.05).sum()
        n += len(res.pairs)
    assert n > 500
    assert 0.02 <= cal / n <= 0.09  # nominal 5 %
    assert naive / n > 0.075 and naive / n > cal / n + 0.02  # best-of-two snooping, uncorrected


def test_estimated_fdr_arithmetic():
    pairs = pd.DataFrame({"p_calibrated": [0.001, 0.004, 0.02, 0.03, 0.2, 0.4, 0.6, 0.9]})
    res = ScreenResult(pairs, {"alpha": 0.05}, 10, 30, len(pairs), 16, 0)
    assert res.discoveries(0.05) == 4 and res.discoveries() == 4
    # expected null discoveries = 0.05 * 8 = 0.4 -> FDR 0.1 ; at 0.01: 0.08 expected / 2 observed
    assert res.estimated_fdr(0.05) == pytest.approx(0.1)
    assert res.estimated_fdr(0.01) == pytest.approx(0.04)
    assert np.isnan(res.estimated_fdr(0.0001))  # no discoveries -> undefined, not zero
    assert res.estimated_fdr(0.9) == pytest.approx(0.9)  # 0.9 * 8 expected / 8 observed
    many = ScreenResult(pairs, {"alpha": 0.05}, 10, 30, 100, 200, 0)  # 100 candidates tested
    assert many.estimated_fdr(0.05) == 1.0  # 5 expected false > 4 observed: capped at 1, not 1.25


def test_summary_records_the_counts_later_corrections_need_and_is_json_serialisable():
    logp, sectors = factor_panel(planted=2, seed=1)
    res = screen_pairs(logp, sectors, ScreenConfig(**FAST))
    s = res.summary()
    assert s["n_tests"] == 2 * s["n_candidates"] and s["n_family"] >= s["n_candidates"]
    assert s["n_eligible"] == logp.shape[1] and len(s["n_null_candidates"]) == FAST["n_null_panels"]
    # the null applies the SAME candidate rule, so it must present a comparable number of candidates
    # (not identical: real neighbour lists are more mutual than random ones, so the unions differ ~20 %)
    assert np.mean(s["n_null_candidates"]) == pytest.approx(s["n_candidates"], rel=0.3)
    json.dumps(s)


def test_cluster_candidate_method_runs_end_to_end():
    logp, sectors = factor_panel(planted=2, seed=1, sector_size=18)
    res = screen_pairs(
        logp, sectors, ScreenConfig(**FAST, candidate_method="cluster", cluster_target_size=6)
    )
    assert res.n_candidates > 0
    assert set(res.pairs["y"]) | set(res.pairs["x"]) <= set(logp.columns)
    assert res.pairs["sector"].ne("mixed").all()  # clusters never cross sectors


def test_diagnostics_columns_and_subperiod_stability_are_reported():
    logp, sectors = factor_panel(planted=2, seed=1)
    row = screen_pairs(logp, sectors, ScreenConfig(**FAST)).pairs.iloc[0]
    for col in (
        "beta_half1",
        "beta_half2",
        "beta_rel_change",
        "adf_p_half1",
        "adf_p_half2",
        "hurst",
        "half_life_lo",
        "half_life_hi",
        "spread_sd",
        "kappa",
        "n_obs",
    ):
        assert col in row.index and not pd.isna(row[col]), col
    assert row["half_life_lo"] <= row["half_life"] <= row["half_life_hi"]


def test_input_validation_and_config():
    logp, sectors = factor_panel()
    bad = logp.copy()
    bad.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        screen_pairs(bad, sectors, ScreenConfig(**FAST))
    with pytest.raises(ValidationError, match="k_neighbourz"):
        ScreenConfig(k_neighbourz=3)
    with pytest.raises(ValidationError):
        ScreenConfig(alpha=1.5)


# ---- pipeline-level: eligibility and causality ---------------------------------------------

TRAIN_START, TRAIN_END = "2015-01-02", "2016-12-30"


def build_pipeline(tmp_path, calendar, future_split=False, mutate_future=None, truncate=False):
    """A store of synthetic tickers + a membership file with sectors, as a DataPipeline."""
    sessions = calendar.sessions("2015-01-02", "2017-12-29")
    n = len(sessions)
    rng = np.random.default_rng(11)
    mkt = np.cumsum(rng.normal(0, 0.01, n))
    logp, sectors = {}, {}
    for i in range(24):
        logp[f"T{i:02d}"] = 4 + rng.uniform(0.6, 1.1) * mkt + np.cumsum(rng.normal(0, 0.015, n))
        sectors[f"T{i:02d}"] = "Tech" if i < 12 else "Health"
    for k in range(3):
        x = 4 + 0.8 * mkt + np.cumsum(rng.normal(0, 0.02, n))
        logp[f"X{k}"], logp[f"Y{k}"] = x, 0.3 + 0.7 * x + simulate_ou(0.08, 0, 0.02, n, rng=rng)
        sectors[f"X{k}"] = sectors[f"Y{k}"] = "Tech"
    extras = {
        "LARGEDIV": "Tech",
        "LOWCOV": "Tech",
        "ILLIQ": "Health",
        "STUB": "Health",
        "GONE": "Health",
        "NOSECTOR": "",
        "LATEDIV": "Tech",
        "LATEJOIN": "Health",
    }
    for name, sec in extras.items():
        logp[name] = 4 + 0.9 * mkt + np.cumsum(rng.normal(0, 0.015, n))
        sectors[name] = sec

    rows = [f"{t},,,{s}" for t, s in sectors.items() if t not in ("GONE", "LATEJOIN")]
    rows.append("LATEJOIN,2016-01-04,,Health")  # joins the index mid-training-window
    rows.append("GONE,,2016-06-01,Health")  # left the index before train_end
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "members.csv").write_text("ticker,start,end,sector\n" + "\n".join(rows) + "\n")
    cfg = DataConfig(
        store_dir=str(tmp_path / "store"),
        source="synthetic",
        start=date(2015, 1, 2),
        refresh=RefreshConfig(request_pause_s=0.0),
        universe=UniverseConfig(
            mode="membership_file", membership_file=str(tmp_path / "members.csv")
        ),
    )
    pipe = DataPipeline(cfg, source=SyntheticSource(SyntheticWorld(tickers=[])), calendar=calendar)
    end_pos = sessions.get_loc(pd.Timestamp(TRAIN_END)) + 1
    for name, lp in logp.items():
        close = np.exp(lp)
        vol = np.full(n, 2_000_000.0)
        if name == "ILLIQ":
            vol[:] = 100.0
        if name == "STUB":
            vol[::2] = 0.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": vol,
            },
            index=pd.DatetimeIndex(sessions, name="date"),
        )
        if name == "LOWCOV":
            df = df.drop(df.index[100:140])
        acts = []
        if name == "LARGEDIV":
            acts.append((sessions[200], "dividend", 0.4 * df["close"].iloc[199]))
        if name == "LATEDIV":
            acts.append(
                (sessions[end_pos + 50], "dividend", 0.4 * close[end_pos + 49])
            )  # after train_end
        if future_split and name == "Y0":
            k = end_pos + 100  # a 2-for-1 split after train_end: vendor history is divided by 2
            df.iloc[:k, df.columns.get_indexer(["open", "high", "low", "close"])] *= 0.5
            df.iloc[:k, df.columns.get_loc("volume")] *= 2.0
            acts.append((sessions[k], "split", 2.0))
        if mutate_future is not None:
            fut = df.index > pd.Timestamp(TRAIN_END)
            noise = np.random.default_rng(mutate_future).lognormal(0, 0.5, fut.sum())
            df.loc[fut, ["open", "high", "low", "close"]] *= noise[:, None]
        if truncate:
            df = df.iloc[:end_pos]
        actions = (
            pd.DataFrame(acts, columns=["date", "kind", "value"])
            if acts
            else pd.DataFrame(
                {
                    "date": pd.Series(dtype="datetime64[ns]"),
                    "kind": pd.Series(dtype=object),
                    "value": pd.Series(dtype=float),
                }
            )
        )
        if len(actions):
            actions["date"] = actions["date"].astype("datetime64[ns]")
            if truncate:
                actions = actions[actions["date"] <= pd.Timestamp(TRAIN_END)]
        pipe.store.write_history(name, df, actions)
    return pipe


CFG = ScreenConfig(n_null_panels=5, k_neighbours=4, seed=7, min_median_dollar_volume=1e7)


@pytest.fixture
def pipe(tmp_path, calendar):
    p = build_pipeline(tmp_path / "base", calendar)
    yield p
    p.close()


def test_eligibility_excludes_each_bad_ticker_for_the_right_reason(pipe):
    res = discover_pairs(pipe, TRAIN_START, TRAIN_END, CFG)
    ex = res.excluded
    assert "coverage" in ex["LOWCOV"]
    assert "dollar volume" in ex["ILLIQ"]
    assert "verified" in ex["STUB"]  # zero-volume half the time: reused-symbol heuristic
    assert "non-ordinary" in ex["LARGEDIV"]
    assert "sector" in ex["NOSECTOR"]
    assert (
        "GONE" not in res.pairs["y"].tolist() + res.pairs["x"].tolist()
    )  # not a member on train_end
    used = set(res.pairs["y"]) | set(res.pairs["x"])
    assert (
        used.isdisjoint(ex) and "LATEDIV" not in ex
    )  # a large dividend AFTER train_end is invisible
    # membership is judged AT train_end: a late joiner is in, a leaver is not even considered
    assert "LATEJOIN" not in ex and "GONE" not in ex
    assert (
        res.n_eligible == 32
    )  # 24 stocks + 3 planted pairs + LATEDIV + LATEJOIN; six bad ones out


def test_discovery_on_the_store_finds_the_planted_pairs(pipe):
    res = discover_pairs(pipe, TRAIN_START, TRAIN_END, CFG)
    sel = {frozenset((r.y, r.x)) for r in res.selected.itertuples()}
    assert sum(frozenset((f"Y{k}", f"X{k}")) in sel for k in range(3)) >= 2
    assert res.train_window == (TRAIN_START, TRAIN_END)
    assert "caveats" in res.meta and res.meta["n_members_train_end"] == 37  # 38 tickers, minus GONE


def test_result_does_not_depend_on_anything_after_train_end(tmp_path, calendar, pipe):
    """Look-ahead test 1: scramble every price after train_end; the discoveries must not change."""
    base = discover_pairs(pipe, TRAIN_START, TRAIN_END, CFG)
    noisy = build_pipeline(tmp_path / "noisy", calendar, mutate_future=5)
    try:
        other = discover_pairs(noisy, TRAIN_START, TRAIN_END, CFG)
    finally:
        noisy.close()
    pd.testing.assert_frame_equal(base.pairs, other.pairs)
    assert base.excluded == other.excluded


def test_result_equals_that_of_a_store_that_ends_at_train_end(tmp_path, calendar, pipe):
    """Look-ahead test 2: a store that stops at train_end (later bars and actions absent) agrees."""
    base = discover_pairs(pipe, TRAIN_START, TRAIN_END, CFG)
    short = build_pipeline(tmp_path / "short", calendar, truncate=True)
    try:
        other = discover_pairs(short, TRAIN_START, TRAIN_END, CFG)
    finally:
        short.close()
    pd.testing.assert_frame_equal(base.pairs, other.pairs)


def test_a_split_after_train_end_does_not_change_the_result(tmp_path, calendar, pipe):
    """Look-ahead test 3: a later split rescales history on the vendor basis; the as-of view undoes it."""
    base = discover_pairs(pipe, TRAIN_START, TRAIN_END, CFG)
    split = build_pipeline(tmp_path / "split", calendar, future_split=True)
    try:
        other = discover_pairs(split, TRAIN_START, TRAIN_END, CFG)
    finally:
        split.close()
    pd.testing.assert_frame_equal(base.pairs, other.pairs, rtol=1e-9, atol=1e-9)


def test_too_few_eligible_tickers_is_an_error(pipe):
    with pytest.raises(ValueError, match="eligible"):
        discover_pairs(pipe, TRAIN_START, TRAIN_END, ScreenConfig(min_median_dollar_volume=1e15))
