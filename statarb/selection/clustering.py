"""Hierarchical clustering of stocks on correlation distance, within sectors.

Used as an alternative candidate generator: clusters supply the *groups* inside which
``correlation.top_k_neighbours`` searches for peers.  Average linkage on Mantegna distance; each
sector is cut into ``ceil(size / target_size)`` clusters so the number of within-group pairs stays
bounded.  Clusters never cross sectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from statarb.selection.correlation import correlation_distance


def cluster_groups(
    corr: pd.DataFrame, sectors: pd.Series, target_size: int = 8, method: str = "average"
) -> pd.Series:
    """Group label ``"<sector>|<cluster id>"`` for every ticker in ``corr``."""
    labels = pd.Series(index=corr.columns, dtype=object)
    for sector, members in sectors.reindex(corr.columns).groupby(sectors.reindex(corr.columns)):
        names = list(members.index)
        if len(names) <= target_size:
            labels[names] = f"{sector}|0"
            continue
        dist = correlation_distance(corr.loc[names, names]).to_numpy().copy()  # views are read-only
        np.fill_diagonal(dist, 0.0)
        z = linkage(squareform((dist + dist.T) / 2, checks=False), method=method)
        n_clusters = int(np.ceil(len(names) / target_size))
        labels[names] = [f"{sector}|{c}" for c in fcluster(z, n_clusters, criterion="maxclust")]
    return labels
