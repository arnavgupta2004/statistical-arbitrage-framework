from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.selection.clustering import cluster_groups
from statarb.selection.correlation import (
    correlation_distance,
    n_group_pairs,
    return_correlation,
    top_k_neighbours,
)


def corr_frame(values, names):
    return pd.DataFrame(np.asarray(values, dtype=float), index=names, columns=names)


def block_panel(n=600, blocks=(6, 6, 6), noise=0.5, seed=0):
    """Log prices whose returns have a strong common factor inside each block, ~none across blocks."""
    rng = np.random.default_rng(seed)
    cols, sectors = {}, {}
    for b, size in enumerate(blocks):
        factor = rng.normal(size=n)
        for i in range(size):
            name = f"B{b}_{i}"
            cols[name] = np.cumsum(factor + noise * rng.normal(size=n)) * 0.01
            sectors[name] = "S"
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(cols, index=idx), pd.Series(sectors)


def test_return_correlation_matches_numpy_and_has_unit_diagonal():
    logp, _ = block_panel()
    c = return_correlation(logp)
    assert np.allclose(np.diag(c), 1.0) and np.allclose(c, c.T)
    a, b = logp.columns[0], logp.columns[1]
    assert c.at[a, b] == pytest.approx(
        np.corrcoef(logp[a].diff().dropna(), logp[b].diff().dropna())[0, 1]
    )


def test_return_correlation_refuses_gaps():
    logp, _ = block_panel()
    logp.iloc[10, 0] = np.nan
    with pytest.raises(ValueError, match="gaps"):
        return_correlation(logp)


def test_correlation_distance_is_a_metric():
    logp, _ = block_panel(blocks=(5, 5))
    d = correlation_distance(return_correlation(logp)).to_numpy()
    assert np.allclose(np.diag(d), 0) and np.allclose(d, d.T) and (d >= 0).all()
    assert (
        d[:, None, :] <= d[:, :, None] + d[None, :, :] + 1e-12
    ).all()  # d(i,k) <= d(i,j) + d(j,k)
    assert correlation_distance(corr_frame([[1, -1], [-1, 1]], list("ab"))).at[
        "a", "b"
    ] == pytest.approx(1.0)
    assert correlation_distance(corr_frame([[1, 1], [1, 1]], list("ab"))).at[
        "a", "b"
    ] == pytest.approx(0.0)


def test_top_k_by_hand():
    c = corr_frame(
        [[1, 0.9, 0.2, 0.1], [0.9, 1, 0.3, 0.8], [0.2, 0.3, 1, 0.5], [0.1, 0.8, 0.5, 1]],
        list("abcd"),
    )
    assert top_k_neighbours(c, None, 1) == [("a", "b"), ("b", "d"), ("c", "d")]
    # k=2 adds each stock's second-best peer; pairs are unique and ordered (a < b)
    got = top_k_neighbours(c, None, 2)
    assert got == sorted(set(got)) and all(a < b for a, b in got)
    assert ("a", "b") in got and ("b", "d") in got and ("c", "d") in got and ("b", "c") in got


def test_neighbours_never_cross_groups():
    logp, _ = block_panel()
    groups = pd.Series({c: c.split("_")[0] for c in logp.columns})
    for a, b in top_k_neighbours(return_correlation(logp), groups, 3):
        assert groups[a] == groups[b]


def test_k_larger_than_group_and_singleton_groups():
    c = corr_frame(np.eye(4) + 0.3 * (1 - np.eye(4)), list("abcd"))
    groups = pd.Series({"a": "g", "b": "g", "c": "h", "d": "i"})
    assert top_k_neighbours(c, groups, 10) == [
        ("a", "b")
    ]  # singletons produce nothing; k is capped


def test_top_k_is_deterministic_under_ties():
    c = corr_frame(np.full((5, 5), 0.5) + 0.5 * np.eye(5), list("abcde"))
    assert top_k_neighbours(c, None, 2) == top_k_neighbours(c.copy(), None, 2)


def test_n_group_pairs_counts_the_family():
    assert n_group_pairs(pd.Series(["a"] * 4 + ["b"] * 3 + ["c"])) == 6 + 3 + 0


def test_clusters_recover_planted_blocks_and_never_cross_sectors():
    logp, sectors = block_panel(blocks=(6, 6, 6))
    labels = cluster_groups(return_correlation(logp), sectors, target_size=6)
    blocks = pd.Series({c: c.split("_")[0] for c in logp.columns})
    # every cluster is inside one planted block, and each block ends up in one cluster
    assert (labels.groupby(labels).apply(lambda g: blocks[g.index].nunique()) == 1).all()
    assert labels.nunique() == 3

    two_sectors = pd.Series({c: ("X" if c.startswith("B0") else "Y") for c in logp.columns})
    lab2 = cluster_groups(return_correlation(logp), two_sectors, target_size=4)
    assert all(lab.split("|")[0] == two_sectors[t] for t, lab in lab2.items())


def test_small_sectors_are_a_single_cluster_and_output_is_deterministic():
    logp, sectors = block_panel(blocks=(4,))
    lab = cluster_groups(return_correlation(logp), sectors, target_size=8)
    assert lab.nunique() == 1
    big, sec = block_panel(blocks=(9, 9))
    c = return_correlation(big)
    pd.testing.assert_series_equal(cluster_groups(c, sec, 5), cluster_groups(c, sec, 5))
