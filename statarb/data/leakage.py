"""Reusable look-ahead detectors: any function of history can be put through these.

Two complementary properties of a *causal* function ``f`` mapping a time-indexed frame to a
time-indexed frame:

**Truncation invariance.**  ``f(data[:t])[:t] == f(data)[:t]`` for every cut ``t``.  If the output
at ``s <= t`` changes when rows after ``t`` are removed, then ``f`` read the future.

**Future-perturbation invariance.**  Replacing every row after ``t`` with garbage must leave
``f(...)[:t]`` unchanged.  This catches leaks truncation cannot, such as a function that peeks at a
row *after* the cut but only when it exists in a way that happens to cancel.

Both are necessary, not sufficient: passing shows no leak was found at the tested cuts.  Each
detector is itself tested against a deliberately leaky function (see ``tests/test_leakage.py``), so
a green result means something.  The backtester (Stage 5) and every signal (Stage 4+) reuse these.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

Frame = pd.DataFrame | pd.Series


@dataclass
class LeakageReport:
    passed: bool
    n_cuts: int
    violations: list[dict] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.passed:
            worst = max(self.violations, key=lambda v: v["max_abs_diff"])
            raise AssertionError(
                f"look-ahead detected at {len(self.violations)}/{self.n_cuts} cuts; "
                f"worst: cut={worst['cut']} ({worst['test']}) max|diff|={worst['max_abs_diff']:.3g}"
            )


def _max_abs_diff(a: Frame, b: Frame) -> float:
    """Max absolute difference on the common index; NaN patterns must match exactly."""
    a, b = a.sort_index(), b.sort_index()
    if not a.index.equals(b.index):
        return float("inf")
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape or not np.array_equal(np.isnan(x), np.isnan(y)):
        return float("inf")
    both = ~np.isnan(x)
    return float(np.max(np.abs(x[both] - y[both]))) if both.any() else 0.0


def truncation_invariance(
    fn: Callable[[Frame], Frame], data: Frame, cuts, atol: float = 1e-12
) -> LeakageReport:
    full = fn(data)
    violations = []
    cuts = list(cuts)
    for cut in cuts:
        cut = pd.Timestamp(cut)
        truncated = fn(data.loc[:cut])
        d = _max_abs_diff(truncated.loc[:cut], full.loc[:cut])
        if d > atol:
            violations.append({"test": "truncation", "cut": cut, "max_abs_diff": d})
    return LeakageReport(not violations, len(cuts), violations)


def future_perturbation_invariance(
    fn: Callable[[Frame], Frame], data: Frame, cuts, seed: int = 0, atol: float = 1e-12
) -> LeakageReport:
    """Overwrite everything after each cut with noise (same shape, positive-valued) and re-run."""
    rng = np.random.default_rng(seed)
    base = fn(data)
    violations = []
    cuts = list(cuts)
    for cut in cuts:
        cut = pd.Timestamp(cut)
        future = data.index > cut
        perturbed = data.copy()
        noise = rng.lognormal(mean=0.0, sigma=1.0, size=perturbed.loc[future].shape)
        perturbed.loc[future] = np.asarray(perturbed.loc[future]) * noise
        d = _max_abs_diff(fn(perturbed).loc[:cut], base.loc[:cut])
        if d > atol:
            violations.append({"test": "future_perturbation", "cut": cut, "max_abs_diff": d})
    return LeakageReport(not violations, len(cuts), violations)


def assert_no_lookahead(
    fn: Callable[[Frame], Frame], data: Frame, cuts, atol: float = 1e-12
) -> None:
    """Run both detectors and raise ``AssertionError`` if either finds a leak."""
    for report in (
        truncation_invariance(fn, data, cuts, atol),
        future_perturbation_invariance(fn, data, cuts, atol=atol),
    ):
        report.raise_if_failed()
