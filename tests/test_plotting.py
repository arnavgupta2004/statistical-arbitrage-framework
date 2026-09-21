"""Smoke tests of the figure helpers: they draw what they are given, and the files are written."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.research import plotting as P


@pytest.fixture(scope="module")
def series():
    idx = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(0)
    return {
        "a": pd.Series(rng.normal(0.001, 0.01, 120), index=idx),
        "b": pd.Series(rng.normal(0, 0.01, 120), index=idx),
    }


def test_equity_and_drawdown_draw_one_line_per_series_and_end_where_the_math_says(series):
    P.apply_style()
    fig, ax = P.plt.subplots()
    P.plot_equity(ax, series)
    lines = [ln for ln in ax.lines if ln.get_label() in series]
    assert len(lines) == 2
    assert lines[0].get_ydata()[-1] == pytest.approx(float(np.prod(1 + series["a"].to_numpy())))
    fig2, ax2 = P.plt.subplots()
    P.plot_drawdown(ax2, series)
    assert all(ln.get_ydata().max() <= 1e-12 for ln in ax2.lines if ln.get_label() in series)
    P.plt.close(fig), P.plt.close(fig2)


def test_dot_plot_heatmap_and_save(tmp_path):
    fig, ax = P.plt.subplots()
    P.dot_plot(ax, ["x", "y", "z"], [1.0, -0.5, 0.2], ["k"] * 3, ref=0.0)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["z", "y", "x"] or len(
        ax.get_yticks()
    ) == 3
    m = pd.DataFrame([[1.0, -2.0], [0.5, 0.0]], index=[3, 5], columns=[1.0, 2.0])
    fig2, ax2 = P.plt.subplots()
    P.heatmap(ax2, m)
    assert len(ax2.texts) == 4 and ax2.texts[0].get_text() == "+1.00"
    out = tmp_path / "f.png"
    P.save(fig, out)
    assert out.exists() and out.stat().st_size > 1000
    P.plt.close(fig2)


def test_colours_follow_the_family():
    assert P.colour_of("pca_reversal_k5") == P.PALETTE["pca_reversal"]
    assert P.colour_of("pca_sscore_k10") == P.PALETTE["pca_sscore"]
    assert P.colour_of("static_in2") == P.PALETTE["pairs"]
