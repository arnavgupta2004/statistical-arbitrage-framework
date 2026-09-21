"""Stage 8 post-hoc diagnostic: why does the Reality Check reject where SPA does not?

Written after the main Stage 8 run, to test an explanation.  The Reality Check is not studentised: it
compares the best strategy's raw mean with a bootstrap null whose noise sits at the family's scale.  If the
best strategy is also the most volatile one it is credited for its own volatility.  Test: rescale every series
to the same volatility (the family's mean volatility) and re-run the Reality Check; if the explanation is right
its p-value moves toward SPA's (which is scale-free and does not change).

Uses the same seed, block length and number of resamples as ``stage8_multiple_testing``.  Descriptive only:
no decision rests on it, and it adds no trial.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage8_rc_scale_check
"""

from __future__ import annotations

import json
import sys

import numpy as np

from experiments.stage8_multiple_testing import (
    MEAN_BLOCK,
    N_BOOT,
    PERIODS,
    SEED,
    R,
    git_commit,
    load_series,
)
from statarb.research.registry import Registry
from statarb.statistics.multiple_testing import reality_check, spa_test


def main() -> None:
    gross, _, pairs_cfg, pca_cfg = load_series()
    out = {
        "experiment": "stage8_rc_scale_check",
        "seed": SEED,
        "git_commit": git_commit(),
        "families": {},
    }
    vol = gross.std() * np.sqrt(PERIODS) * 100
    out["annual_vol_pct"] = {c: float(v) for c, v in vol.items()}
    for name, cols in (("all_15", list(gross.columns)), ("pca_6", pca_cfg), ("pairs_9", pairs_cfg)):
        x = gross[cols].to_numpy()
        equal = x / x.std(axis=0) * x.std(axis=0).mean()
        out["families"][name] = {
            "reality_check_p": reality_check(x, N_BOOT, MEAN_BLOCK, SEED)["p_value"],
            "reality_check_p_equal_volatility": reality_check(equal, N_BOOT, MEAN_BLOCK, SEED)[
                "p_value"
            ],
            "spa_consistent_p": spa_test(x, N_BOOT, MEAN_BLOCK, SEED)["consistent"],
            "spa_consistent_p_equal_volatility": spa_test(equal, N_BOOT, MEAN_BLOCK, SEED)[
                "consistent"
            ],
            "vol_pct_min_median_max": [
                float(vol[cols].min()),
                float(vol[cols].median()),
                float(vol[cols].max()),
            ],
            "best_strategy_vol_pct": float(vol[gross[cols].mean().idxmax()]),
        }
        f = out["families"][name]
        print(
            f"{name}: RC {f['reality_check_p']:.3f} -> {f['reality_check_p_equal_volatility']:.3f} with equal volatility; "
            f"SPA {f['spa_consistent_p']:.3f} -> {f['spa_consistent_p_equal_volatility']:.3f}; best strategy vol {f['best_strategy_vol_pct']:.2f} %"
        )
    (R / "stage8_rc_scale_check.json").write_text(json.dumps(out, indent=2))
    Registry().register(
        stage=8,
        kind="diagnostic",
        name="reality_check_scale_dependence_posthoc",
        phases=["research"],
        strategy="none (post-hoc explanation of RC vs SPA)",
        parameters={"n_boot": N_BOOT, "mean_block": MEAN_BLOCK},
        seed=SEED,
        git_commit=git_commit(),
        notes="post-hoc, descriptive; adds no trial; research phase only",
    )


if __name__ == "__main__":
    sys.exit(main())
