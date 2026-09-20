from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.backtest.pair_pnl import pair_returns, run_pair
from statarb.models.hedge_ratio import HedgeSpec, hedge_path
from statarb.portfolio.construction import entry_notional, spread_returns
from statarb.signals.pairs import PairParams, generate_positions
from statarb.signals.zscore import spread_zscore
from statarb.statistics.bootstrap import bootstrap_ci, sharpe, stationary_bootstrap_indices

from .test_hedge_ratio import pair


def test_indices_are_valid_and_blocks_have_the_requested_mean_length():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(50_000, 10.0, rng)
    assert idx.min() >= 0 and idx.max() < 50_000
    consecutive = (np.diff(idx) % 50_000) == 1
    assert 1 / (1 - consecutive.mean()) == pytest.approx(10.0, rel=0.05)  # mean block length


def test_a_block_length_of_one_is_the_iid_bootstrap():
    idx = stationary_bootstrap_indices(20_000, 1.0, np.random.default_rng(1))
    assert ((np.diff(idx) % 20_000) == 1).mean() < 0.01


def test_blocks_preserve_serial_dependence_that_an_iid_bootstrap_destroys():
    rng = np.random.default_rng(2)
    x = np.zeros(4000)
    for t in range(1, 4000):
        x[t] = 0.8 * x[t - 1] + rng.normal()
    ac = lambda v: np.corrcoef(v[:-1], v[1:])[0, 1]  # noqa: E731
    block = np.mean([ac(x[stationary_bootstrap_indices(4000, 20, rng)]) for _ in range(30)])
    iid = np.mean([ac(x[stationary_bootstrap_indices(4000, 1, rng)]) for _ in range(30)])
    assert block > 0.6 and abs(iid) < 0.05


def test_bootstrap_interval_for_the_mean_covers_at_about_the_nominal_rate():
    rng = np.random.default_rng(3)
    cover = 0
    for _ in range(120):
        x = rng.normal(0.5, 1.0, 300)
        _, (lo, hi), _ = bootstrap_ci(
            x, np.mean, n_boot=300, mean_block=5, seed=int(rng.integers(1e9))
        )
        cover += lo <= 0.5 <= hi
    assert 0.88 <= cover / 120 <= 0.99


def test_dependent_data_gives_a_wider_interval_with_blocks_than_iid():
    rng = np.random.default_rng(4)
    x = np.zeros(1500)
    for t in range(1, 1500):
        x[t] = 0.7 * x[t - 1] + rng.normal()
    _, (l1, h1), _ = bootstrap_ci(x, np.mean, 800, mean_block=1, seed=1)
    _, (l2, h2), _ = bootstrap_ci(x, np.mean, 800, mean_block=25, seed=1)
    assert (h2 - l2) > 1.5 * (h1 - l1)


def test_sharpe_and_argument_validation():
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.01])
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert np.isnan(sharpe(np.zeros(10)))
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(1, 5, np.random.default_rng(0))


def test_run_pair_is_exactly_the_manual_chain():
    ly, lx, _ = pair(900, seed=12, drift_sd=0.001)
    ry, rx = ly.diff().fillna(0.0), lx.diff().fillna(0.0)
    spec, params, train_end = HedgeSpec("rolling", window=90), PairParams(), ly.index[499]
    got = run_pair(ly, lx, ry, rx, spec, 60, params, train_end, target_vol=0.01, freeze_beta=True)

    beta = hedge_path(ly, lx, spec, train_end)["beta"]
    z = spread_zscore(ly, lx, beta, 60)["z"]
    res = generate_positions(z.to_numpy(), params, trade_start=500)
    n = entry_notional(res.position, spread_returns(ry, rx, beta), 0.01)
    want = pair_returns(res.position, beta, n, ry, rx, freeze_beta=True)
    pd.testing.assert_frame_equal(got["returns"], want)
    assert (got["returns"]["pnl"].iloc[:500] == 0).all() and len(got["trades"]) > 0
    assert set(got) == {"returns", "z", "beta", "positions", "trades"}
