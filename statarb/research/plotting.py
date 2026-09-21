"""Figure helpers for the final report: one consistent style and a few reusable primitives.

Nothing here computes research results; figures are composed from saved results in
``experiments/stage10_figures.py``.  The Agg backend is forced so figures render headless.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from statarb.backtest import metrics  # noqa: E402

PALETTE = {
    "pca_reversal": "#1b6ca8",
    "pca_sscore": "#d98c1f",
    "pairs": "#5b8f29",
    "placebo": "#9aa0a6",
    "bad": "#b03a2e",
    "ink": "#22262b",
    "muted": "#6b7280",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "axes.prop_cycle": matplotlib.cycler(color=list(PALETTE.values())[:5]),
        }
    )


def year_axis(*axes) -> None:
    """One tick per year, labelled with the year only (default date ticks collide on wide spans)."""
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def colour_of(label: str) -> str:
    if label.startswith("pca_reversal"):
        return PALETTE["pca_reversal"]
    if label.startswith("pca_sscore"):
        return PALETTE["pca_sscore"]
    return PALETTE["pairs"]


def plot_equity(ax, series: dict[str, pd.Series], colours: dict[str, str] | None = None) -> None:
    """Cumulative growth of 1 for each daily return series (compounded)."""
    for label, r in series.items():
        eq = pd.Series(metrics.equity_curve(r.to_numpy()), index=r.index)
        ax.plot(eq.index, eq.values, lw=1.4, label=label, color=(colours or {}).get(label))
    ax.axhline(1.0, color=PALETTE["muted"], lw=0.8)
    ax.set_ylabel("growth of 1")


def plot_drawdown(ax, series: dict[str, pd.Series], colours: dict[str, str] | None = None) -> None:
    for label, r in series.items():
        dd = pd.Series(metrics.drawdowns(r.to_numpy()), index=r.index) * 100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.25, color=(colours or {}).get(label))
        ax.plot(dd.index, dd.values, lw=1.0, label=label, color=(colours or {}).get(label))
    ax.set_ylabel("drawdown, %")


def dot_plot(
    ax, labels: list[str], values: list[float], colours: list[str], ref: float | None = None
):
    """Horizontal dot plot, first label at the top."""
    y = np.arange(len(labels))[::-1]
    ax.scatter(values, y, c=colours, s=22, zorder=3)
    ax.set_yticks(y, labels)
    if ref is not None:
        ax.axvline(ref, color=PALETTE["ink"], lw=0.9, ls="--")
    ax.grid(axis="y", alpha=0.4)


def heatmap(
    ax, matrix: pd.DataFrame, fmt: str = "{:+.2f}", cmap: str = "RdBu", centre: float = 0.0
):
    vmax = float(np.nanmax(np.abs(matrix.to_numpy() - centre)))
    im = ax.imshow(
        matrix.to_numpy(), cmap=cmap, vmin=centre - vmax, vmax=centre + vmax, aspect="auto"
    )
    ax.set_xticks(range(matrix.shape[1]), [f"{c:g}" for c in matrix.columns])
    ax.set_yticks(range(matrix.shape[0]), [f"{i:g}" for i in matrix.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, fmt.format(matrix.iloc[i, j]), ha="center", va="center", fontsize=8)
    ax.grid(False)
    return im


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)
