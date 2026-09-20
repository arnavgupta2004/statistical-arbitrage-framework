"""Return correlation, correlation distance, and rank-based candidate neighbours.

Candidate generation is deliberately **rank-based** (each stock's ``k`` most correlated peers) and
never uses an absolute correlation threshold.  The calibration null (``screening.py``) is built by
applying the *same* generator to panels whose stocks are independent by construction; a threshold
such as "corr > 0.5" could never be met there, so the null would be empty and the calibration void.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def return_correlation(logp: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of daily log returns.  ``logp`` must be gap-free (no imputation here)."""
    r = logp.diff().iloc[1:]
    if r.isna().any().any():
        raise ValueError("log prices contain gaps; align and drop incomplete tickers first")
    c = np.corrcoef(r.to_numpy(), rowvar=False)
    return pd.DataFrame(c, index=logp.columns, columns=logp.columns)


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Mantegna's metric ``d = sqrt((1 - rho) / 2)`` (a true metric on correlation matrices)."""
    return np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))


def top_k_neighbours(corr: pd.DataFrame, groups: pd.Series | None, k: int) -> list[tuple[str, str]]:
    """Unique unordered pairs ``(a, b)``, ``a < b``: each stock's ``k`` best-correlated peers.

    Peers are searched only inside the stock's group (sector or cluster) when ``groups`` is given.
    Ties break by column order, so the result is deterministic.
    """
    names = list(corr.columns)
    if groups is None:
        buckets = {"all": names}
    else:
        buckets = {}
        for name in names:
            buckets.setdefault(groups[name], []).append(name)
    values = corr.to_numpy()
    pos = {n: i for i, n in enumerate(names)}
    pairs: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        idx = np.array([pos[m] for m in members])
        sub = values[np.ix_(idx, idx)].copy()
        np.fill_diagonal(sub, -np.inf)
        kk = min(k, len(members) - 1)
        order = np.argsort(-sub, axis=1, kind="stable")[:, :kk]
        for i, row in enumerate(order):
            for j in row:
                a, b = members[i], members[int(j)]
                pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


def n_group_pairs(groups: pd.Series) -> int:
    """Number of same-group pairs: the full *family* a screen could have tested."""
    sizes = groups.value_counts().to_numpy()
    return int((sizes * (sizes - 1) // 2).sum())
