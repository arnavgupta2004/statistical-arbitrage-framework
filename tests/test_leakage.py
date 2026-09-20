"""The look-ahead detectors, validated against functions known to be causal and known to leak.

A detector that has never been shown to fail proves nothing, so half of this file is deliberately
leaky code that the detectors must catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.data.leakage import (
    LeakageReport,
    assert_no_lookahead,
    future_perturbation_invariance,
    truncation_invariance,
)


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=300)
    return pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (300, 3)), axis=0)),
        index=idx,
        columns=list("abc"),
    )


@pytest.fixture
def cuts(prices):
    return prices.index[[30, 80, 150, 220, 290]]


CAUSAL = {
    "rolling_mean": lambda x: x.rolling(20).mean(),
    "pct_change": lambda x: x.pct_change(),
    "log_diff": lambda x: np.log(x).diff(5),
    "ewm": lambda x: x.ewm(span=10).mean(),
    "lag": lambda x: x.shift(1),
    "expanding_zscore": lambda x: (x - x.expanding().mean()) / x.expanding().std(),
    "rolling_zscore": lambda x: (x - x.rolling(30).mean()) / x.rolling(30).std(),
    "cummax_drawdown": lambda x: x / x.cummax() - 1,
}

LEAKY = {
    "lead_by_one": lambda x: x.shift(-1),
    "centered_window": lambda x: x.rolling(5, center=True).mean(),
    "full_sample_zscore": lambda x: (x - x.mean()) / x.std(),
    "normalise_by_last_row": lambda x: x / x.iloc[-1],
    "future_shifted_average": lambda x: x.rolling(20).mean().shift(-3),
    "full_sample_max": lambda x: x / x.max(),
    "full_sample_demeaned_returns": lambda x: x.pct_change() - x.pct_change().mean(),
}


@pytest.mark.parametrize("name", CAUSAL)
def test_causal_functions_pass_both_detectors(prices, cuts, name):
    fn = CAUSAL[name]
    assert truncation_invariance(fn, prices, cuts).passed
    assert future_perturbation_invariance(fn, prices, cuts).passed
    assert_no_lookahead(fn, prices, cuts)


@pytest.mark.parametrize("name", LEAKY)
def test_leaky_functions_are_caught_by_both_detectors(prices, cuts, name):
    fn = LEAKY[name]
    assert not truncation_invariance(fn, prices, cuts).passed
    assert not future_perturbation_invariance(fn, prices, cuts).passed
    with pytest.raises(AssertionError, match="look-ahead detected"):
        assert_no_lookahead(fn, prices, cuts)


def test_a_leak_confined_to_one_column_and_one_late_window_is_still_found(prices, cuts):
    def sneaky(x):
        out = x.pct_change()
        out.loc[x.index[200:], "c"] = (
            x["c"].shift(-1).loc[x.index[200:]]
        )  # peeks only late, in one column
        return out

    report = truncation_invariance(sneaky, prices, cuts)
    assert not report.passed
    assert {v["cut"] for v in report.violations} <= set(
        cuts[3:]
    )  # only cuts at/after the leak window


def test_report_names_the_worst_violation(prices, cuts):
    with pytest.raises(AssertionError, match=r"worst: cut=.*max\|diff\|"):
        truncation_invariance(LEAKY["lead_by_one"], prices, cuts).raise_if_failed()
    LeakageReport(True, 0).raise_if_failed()  # a passing report does nothing


def test_series_input_is_supported(prices, cuts):
    s = prices["a"]
    assert truncation_invariance(lambda x: x.rolling(10).mean(), s, cuts).passed
    assert not truncation_invariance(lambda x: x.shift(-2), s, cuts).passed


def test_detectors_do_not_mutate_the_input(prices, cuts):
    before = prices.copy()
    future_perturbation_invariance(CAUSAL["ewm"], prices, cuts)
    pd.testing.assert_frame_equal(prices, before)
