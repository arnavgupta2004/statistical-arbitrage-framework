"""A small multi-year synthetic store for walk-forward tests (planted cointegrated pairs)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from statarb.config import DataConfig, RefreshConfig, SplitConfig, UniverseConfig
from statarb.data.pipeline import DataPipeline
from statarb.data.sources.base import empty_actions
from statarb.data.sources.synthetic import SyntheticSource, SyntheticWorld
from statarb.models.ou import simulate_ou
from statarb.selection.screening import ScreenConfig

SPLIT = dict(
    data_start=date(2011, 1, 3),
    research_end=date(2013, 12, 31),
    validation_start=date(2014, 1, 1),
    validation_end=date(2014, 12, 31),
    holdout_start=date(2015, 1, 1),
)
SCREEN = ScreenConfig(n_null_panels=3, k_neighbours=4, seed=5, min_median_dollar_volume=1e7)
EVENT_DATE = "2013-06-12"  # the non-ordinary distribution, inside the 2013 test block


def build(tmp_path, calendar, mutate_after=None, mutate_seed=0, truncate_after=None, split=True):
    """Pipeline over 2011-2016: 24 noise stocks, 4 planted pairs (Tech), one pair with a 40 % dividend."""
    sessions = calendar.sessions("2011-01-03", "2016-12-30")
    n = len(sessions)
    rng = np.random.default_rng(21)
    mkt = np.cumsum(rng.normal(0, 0.01, n))
    logp, sector = {}, {}
    for i in range(24):
        logp[f"T{i:02d}"] = 4 + rng.uniform(0.6, 1.1) * mkt + np.cumsum(rng.normal(0, 0.015, n))
        sector[f"T{i:02d}"] = "Tech" if i < 12 else "Health"
    for k in range(4):
        x = 4 + 0.8 * mkt + np.cumsum(rng.normal(0, 0.02, n))
        logp[f"X{k}"], logp[f"Y{k}"] = x, 0.3 + 0.7 * x + simulate_ou(0.08, 0, 0.02, n, rng=rng)
        sector[f"X{k}"] = sector[f"Y{k}"] = "Tech"
    x = 4 + 0.8 * mkt + np.cumsum(rng.normal(0, 0.02, n))
    logp["EVX"], logp["EVY"] = x, 0.3 + 0.7 * x + simulate_ou(0.08, 0, 0.02, n, rng=rng)
    sector["EVX"] = sector["EVY"] = "Tech"

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "members.csv").write_text(
        "ticker,start,end,sector\n" + "\n".join(f"{t},,,{s}" for t, s in sector.items()) + "\n"
    )
    cfg = DataConfig(
        store_dir=str(tmp_path / "store"),
        source="synthetic",
        start=date(2011, 1, 3),
        refresh=RefreshConfig(request_pause_s=0.0),
        universe=UniverseConfig(
            mode="membership_file", membership_file=str(tmp_path / "members.csv")
        ),
        split=SplitConfig(**SPLIT, ledger_path=str(tmp_path / "ledger.jsonl")) if split else None,
    )
    pipe = DataPipeline(cfg, source=SyntheticSource(SyntheticWorld(tickers=[])), calendar=calendar)
    for name, lp in logp.items():
        close = np.exp(lp)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": np.full(n, 2_000_000.0),
            },
            index=pd.DatetimeIndex(sessions, name="date"),
        )
        acts = empty_actions()
        if name == "EVY":
            pos = sessions.get_loc(pd.Timestamp(EVENT_DATE))
            acts = pd.DataFrame(
                {
                    "date": [pd.Timestamp(EVENT_DATE)],
                    "kind": ["dividend"],
                    "value": [0.4 * float(close[pos - 1])],
                }
            )
            acts["date"] = acts["date"].astype("datetime64[ns]")
        if mutate_after is not None:
            fut = df.index > pd.Timestamp(mutate_after)
            noise = np.random.default_rng(mutate_seed).lognormal(0, 0.4, fut.sum())
            df.loc[fut, ["open", "high", "low", "close"]] *= noise[:, None]
        if truncate_after is not None:
            df = df[df.index <= pd.Timestamp(truncate_after)]
            acts = acts[acts["date"] <= pd.Timestamp(truncate_after)] if len(acts) else acts
        pipe.store.write_history(name, df, acts)
    return pipe
