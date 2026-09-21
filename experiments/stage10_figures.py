"""Stage 10: the final report's figures (spec section 24), generated from saved research-phase results.

Every figure is composed from ``experiments/results/*.json|csv`` -- nothing is re-tuned -- plus three things that need
the price store (research-phase dates only, <= 2016 for the pair examples): the representative pair panels, and the
Stage 7 placebo distribution, which is re-derived with Stage 7's seeds and *asserted* to reproduce its saved median.

Selection of what is shown (fixed here, so the display cannot be cherry-picked):
    * Pair examples: rank 1 of the Stage 3 screen (the most significant pair in training) and the pair with the best
      out-of-sample persistence; failures: DHR/HSIC (the spin-off) and AMAT/MU (the largest post-formation mean shift
      among the 20 after DHR/HSIC).  Failures are shown next to successes on purpose.
    * Strategy curves: the best-gross configuration of each family and the pair strategy's a-priori default.  The
      figure captions say so.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.stage10_figures
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from statarb.backtest import metrics
from statarb.backtest.capacity import break_even_capital, net_at_capital, participation_breach_share
from statarb.research import plotting as P
from statarb.research.plotting import PALETTE, plt
from statarb.statistics.bootstrap import bootstrap_ci
from statarb.statistics.sharpe import expected_max_sharpe, min_backtest_years

R = Path("experiments/results")
FIG = Path("docs/figures")
DAYS_PER_YEAR = 252


def J(name: str) -> dict:
    return json.loads((R / name).read_text())


def load_frames() -> dict:
    s5 = pd.read_csv(R / "stage5_daily_returns.csv", parse_dates=["date"])
    s6 = pd.read_csv(R / "stage6_daily_net_returns.csv", parse_dates=["date"])
    s7 = pd.read_csv(R / "stage7_daily_returns.csv", parse_dates=["date"])
    out = {
        "s5": {c: g.set_index("date") for c, g in s5.groupby("config")},
        "s6": {c: g.set_index("date") for c, g in s6.groupby("config")},
        "s7": {c: g.set_index("date") for c, g in s7.groupby("config")},
    }
    return out


def note(ax, text, loc=(0.02, 0.96), **kw):
    ax.text(
        *loc, text, transform=ax.transAxes, va="top", fontsize=7.5, color=PALETTE["muted"], **kw
    )


# ---- pair research ------------------------------------------------------------------------------
def fig_pair_screen(s8: dict) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.6), sharey=True)
    for ax, (name, v) in zip(axes, s8["screen_fdr"].items(), strict=True):
        dens = np.array(v["p_histogram_deciles"]) / (v["n_p_values"] / 10)
        ax.bar(np.arange(10) * 0.1 + 0.05, dens, width=0.09, color=PALETTE["pairs"], alpha=0.85)
        ax.axhline(1.0, color=PALETTE["ink"], ls="--", lw=0.9)
        ax.set_title(
            name.replace("stage3_2011_2015", "Stage 3 window").replace("fold_test_", "test "),
            fontsize=8.5,
        )
        ax.set_xlabel("calibrated p")
        note(
            ax,
            f"p<=0.05: {v['calibrated_discoveries_at_5pct_uncorrected']} (null: {v['expected_under_null_at_5pct']:.0f})\n"
            f"BH q=0.05: {v['bh_q0.05']}   pi0 {v['storey_pi0']['0.5']:.2f}",
            (0.03, 0.97),
        )
    axes[0].set_ylabel("density relative to uniform")
    fig.suptitle(
        "Calibrated cointegration p-values of every candidate pair: no excess of small p over the null",
        y=1.04,
        fontsize=10,
        fontweight="bold",
    )
    P.save(fig, FIG / "01_pair_screen_pvalues.png")


def pair_frames(pipe, y: str, x: str, beta: float, train_end: str, end: str) -> dict:
    from statarb.signals.pairs import PairParams, generate_positions
    from statarb.signals.zscore import spread_zscore

    panel = pipe.panel([y, x], "2011-01-03", end)
    lp = np.log(panel.tr_close())
    ly, lx = lp[y], lp[x]
    n_train = int((lp.index <= pd.Timestamp(train_end)).sum())
    spread = ly - beta * lx
    centre, sd = float(spread.iloc[:n_train].mean()), float(spread.iloc[:n_train].std(ddof=1))
    z = spread_zscore(ly, lx, beta, 60)["z"]
    pos = generate_positions(z.to_numpy(), PairParams(2.0, 0.5, 4.0, 60), trade_start=n_train)
    roll_beta = ly.rolling(250).cov(lx) / lx.rolling(250).var()
    return {
        "idx": lp.index,
        "ly": ly,
        "lx": lx,
        "spread": spread - centre,
        "sd": sd,
        "z": z,
        "pos": pos.position,
        "roll_beta": roll_beta,
        "beta": beta,
        "n_train": n_train,
    }


def draw_pair(
    axes,
    f: dict,
    y: str,
    x: str,
    title: str,
    subtitle: str,
    event: str | None = None,
    zclip: float = 8.0,
) -> None:
    idx, n = f["idx"], f["n_train"]
    t0 = idx[n]
    a, b, c, d = axes
    a.plot(idx, f["ly"] - f["ly"].iloc[0], color=PALETTE["pca_reversal"], lw=1, label=y)
    a.plot(idx, f["lx"] - f["lx"].iloc[0], color=PALETTE["pca_sscore"], lw=1, label=x)
    a.legend(loc="upper left")
    a.set_ylabel("log price (rebased)")
    a.set_title(title + "\n" + subtitle, loc="left", fontsize=8.5)
    P.year_axis(*axes)
    b.plot(idx, f["spread"] / f["sd"], color=PALETTE["ink"], lw=0.9)
    b.axhspan(-2, 2, color=PALETTE["placebo"], alpha=0.2)
    b.set_ylabel("spread, training sd")
    c.plot(idx, f["z"].clip(-zclip, zclip), color=PALETTE["ink"], lw=0.8)
    for lvl, ls in ((2, "--"), (-2, "--"), (0.5, ":"), (-0.5, ":")):
        c.axhline(lvl, color=PALETTE["muted"], lw=0.7, ls=ls)
    pos = pd.Series(f["pos"], index=idx)
    c.fill_between(
        idx, -zclip, zclip, where=pos > 0, color=PALETTE["pca_reversal"], alpha=0.18, step="mid"
    )
    c.fill_between(idx, -zclip, zclip, where=pos < 0, color=PALETTE["bad"], alpha=0.18, step="mid")
    c.set_ylabel("60-day z-score (clipped)")
    d.plot(idx, f["roll_beta"], color=PALETTE["pairs"], lw=1.1, label="rolling 250-day OLS")
    d.axhline(
        f["beta"], color=PALETTE["ink"], ls="--", lw=1, label=f"training beta {f['beta']:.2f}"
    )
    d.set_ylabel("hedge ratio")
    d.legend(loc="best")
    for ax in axes:
        ax.axvline(t0, color=PALETTE["bad"], lw=0.9)
        ax.set_xlim(idx[0], idx[-1])
        if event:
            ax.axvline(pd.Timestamp(event), color=PALETTE["ink"], lw=0.9, ls="-.")
    a.text(t0, a.get_ylim()[1], " test year ->", color=PALETTE["bad"], fontsize=7.5, va="top")


def fig_pairs(
    pipe, sel: pd.DataFrame, pairs: list[tuple[str, str, str, str]], name: str, suptitle: str
) -> None:
    fig, axes = plt.subplots(4, len(pairs), figsize=(5.6 * len(pairs), 8.6), sharex="col")
    axes = axes.reshape(4, len(pairs))
    for j, (y, x, title, event) in enumerate(pairs):
        r = sel[(sel.y == y) & (sel.x == x)].iloc[0]
        f = pair_frames(pipe, y, x, float(r.beta), "2015-12-31", "2016-12-30")
        sub = (
            f"in-sample half-life {r.half_life:.0f}d; test year: spread sd x{r.oos_sd_ratio:.2f}, "
            f"mean shift {r.oos_mean_shift_sd:.1f} sd, ADF p {r.oos_adf_p:.2f}"
        )
        draw_pair(axes[:, j], f, y, x, title, sub, event or None)
    fig.suptitle(suptitle, y=0.995, fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    P.save(fig, FIG / name)


def fig_oos_persistence(sel: pd.DataFrame, s9: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ok = sel.dropna(subset=["oos_half_life"])
    a.scatter(
        ok.half_life,
        ok.oos_half_life.clip(upper=2000),
        color=PALETTE["pairs"],
        s=26,
        zorder=3,
        label="OOS half-life",
    )
    bad = sel[sel.oos_half_life.isna()]
    a.scatter(
        bad.half_life,
        np.full(len(bad), 2200),
        marker="x",
        color=PALETTE["bad"],
        s=30,
        label="no reversion (AR(1) b >= 1)",
    )
    a.plot([5, 60], [5, 60], color=PALETTE["ink"], ls="--", lw=0.9, label="same as in-sample")
    a.set_xscale("log"), a.set_yscale("log")
    a.set_xticks([10, 20, 30, 50], ["10", "20", "30", "50"]), a.minorticks_off()
    a.set_yticks([10, 100, 1000], ["10", "100", "1000"])
    a.set_xlabel("in-sample half-life (days)"), a.set_ylabel("out-of-sample half-life (days)")
    a.set_title("Selected pairs, 2011-15 -> 2016", loc="left")
    a.legend(loc="lower right")
    pool = s9["relationship_stability"]["pooled"]
    labels, rates, lo, hi = [], [], [], []
    for k, nm in (
        ("selected", "selected\n(80 pair-years)"),
        ("baseline_pool", "baseline pool\n(4,807 pair-years)"),
    ):
        labels.append(nm), rates.append(100 * pool[k]["share_oos_adf_rejects_5pct"])
        (
            lo.append(100 * (pool[k]["share_oos_adf_rejects_5pct"] - pool[k]["wilson95"][0])),
            hi.append(100 * (pool[k]["wilson95"][1] - pool[k]["share_oos_adf_rejects_5pct"])),
        )
    b.bar(
        labels,
        rates,
        color=[PALETTE["pairs"], PALETTE["placebo"]],
        yerr=[lo, hi],
        capsize=4,
        width=0.55,
    )
    b.axhline(5, color=PALETTE["ink"], ls="--", lw=0.9)
    b.set_ylabel("out-of-sample ADF rejections at 5 % (%)")
    b.set_title("Do selected pairs stay cointegrated?", loc="left")
    note(
        b,
        "Spearman(train p, OOS p) = "
        f"{pool['rank_correlation_train_p_vs_oos_adf_p']['spearman']:+.3f}",
        (0.03, 0.95),
    )
    fig.tight_layout()
    P.save(fig, FIG / "04_pair_oos_persistence.png")


# ---- backtests ----------------------------------------------------------------------------------
def show_series(fr: dict) -> tuple[dict, dict, dict]:
    """(gross, net, colours) for: best PCA reversal, best PCA s-score, the best-gross pair config, the pair default."""
    gross, net, col = {}, {}, {}
    for label, key, c in (
        ("PCA reversal k=5", ("s7", "pca_reversal_k5"), PALETTE["pca_reversal"]),
        ("PCA s-score k=5", ("s7", "pca_sscore_k5"), PALETTE["pca_sscore"]),
        ("pairs: expanding, entry 1.5 (best gross)", ("s6", "expanding_in1.5"), PALETTE["pairs"]),
        ("pairs: static, entry 2.0 (default)", ("s6", "static_in2"), PALETTE["muted"]),
    ):
        d = fr[key[0]][key[1]]
        gross[label], net[label], col[label] = d["pnl"], d["net"], c
    return gross, net, col


def fig_equity(fr: dict) -> None:
    g, n, col = show_series(fr)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    P.plot_equity(a, g, col), P.plot_equity(b, n, col)
    a.set_title("Gross of costs (research phase 2015-2018)", loc="left")
    b.set_title("Net of costs ($100M, central cost model)", loc="left")
    a.legend(loc="upper left")
    P.year_axis(a, b)
    fig.tight_layout()
    P.save(fig, FIG / "05_equity_curves.png")


def fig_risk(fr: dict) -> None:
    g, n, col = show_series(fr)
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.2), sharex=True)
    for j, (nm, ser) in enumerate((("gross", g), ("net", n))):
        P.plot_drawdown(axes[0, j], ser, col)
        for lab, r in ser.items():
            axes[1, j].plot(metrics.rolling_sharpe(r, 126), color=col[lab], lw=1.1)
            axes[2, j].plot(metrics.rolling_volatility(r, 63) * 100, color=col[lab], lw=1.1)
        axes[1, j].axhline(0, color=PALETTE["ink"], lw=0.8)
        axes[0, j].set_title(f"{nm}", loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    P.year_axis(*axes[2])
    axes[1, 0].set_ylabel("rolling Sharpe (126d)"), axes[2, 0].set_ylabel("rolling vol, % (63d)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    P.save(fig, FIG / "06_drawdown_rolling.png")


def fig_turnover_exposure(fr: dict) -> None:
    rev, sc, pdef = fr["s7"]["pca_reversal_k5"], fr["s7"]["pca_sscore_k5"], fr["s5"]["static_in2"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    a, b, c, d = axes.ravel()
    for lab, df, col in (
        ("PCA reversal k=5", rev, PALETTE["pca_reversal"]),
        ("PCA s-score k=5", sc, PALETTE["pca_sscore"]),
        ("pairs (default)", pdef, PALETTE["pairs"]),
    ):
        a.plot(df["trade"].rolling(21).mean(), color=col, lw=1.2, label=lab)
    (
        a.set_yscale("log"),
        a.set_ylabel("daily turnover, x capital (21d mean)"),
        a.legend(loc="center"),
    )
    a.set_title("Turnover", loc="left")
    b.plot(rev["gross_exposure"], color=PALETTE["pca_reversal"], lw=1, label="gross exposure")
    b.plot(
        rev["net_exposure"],
        color=PALETTE["bad"],
        lw=1,
        label="net exposure (neutral by construction)",
    )
    b.set_ylabel("x capital"), b.legend(), b.set_title("PCA reversal k=5: exposure", loc="left")
    c.plot(rev["n_signal"], color=PALETTE["pca_reversal"], lw=1, label="signal names")
    c.plot(rev["n_names"], color=PALETTE["placebo"], lw=1, label="names touched incl. hedge")
    (
        c.set_ylabel("names"),
        c.legend(loc="center right"),
        c.set_title("PCA reversal k=5: positions", loc="left"),
    )
    d.plot(pdef["n_open"], color=PALETTE["pairs"], lw=1)
    (
        d.set_ylabel("open pairs (of 20 slots)"),
        d.set_title("Pairs (default): open positions", loc="left"),
    )
    P.year_axis(c, d)
    fig.tight_layout()
    P.save(fig, FIG / "07_turnover_exposure.png")


# ---- robustness ---------------------------------------------------------------------------------
def fig_cost(fr: dict) -> None:
    names = [
        ("PCA reversal k=5", "s7", "pca_reversal_k5"),
        ("PCA reversal k=10", "s7", "pca_reversal_k10"),
        ("PCA s-score k=5", "s7", "pca_sscore_k5"),
        ("PCA s-score k=10", "s7", "pca_sscore_k10"),
        ("pairs: expanding 1.5", "s6", "expanding_in1.5"),
        ("pairs: static 2.0", "s6", "static_in2"),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    mults = np.linspace(0, 2, 41)
    ends = []
    for lab, s, c in names:
        d = fr[s][c]
        a.plot(
            mults,
            [metrics.sharpe((d["pnl"] - m * d["cost"]).to_numpy()) for m in mults],
            color=P.colour_of(c),
            lw=1.3,
            ls="-" if s == "s7" else "--",
        )
        ends.append((metrics.sharpe((d["pnl"] - 2 * d["cost"]).to_numpy()), lab, P.colour_of(c)))
    ends.sort()
    ypos = []
    for e, _, _ in ends:  # spread the end labels so none overprint
        ypos.append(e if not ypos else max(e, ypos[-1] + 0.55))
    for (_, lab, col), yy in zip(ends, ypos, strict=True):
        a.text(2.03, yy, lab, fontsize=6.8, va="center", color=col)
    a.axhline(0, color=PALETTE["ink"], lw=0.9), a.axvline(1, color=PALETTE["muted"], lw=0.8, ls=":")
    (
        a.set_xlim(0, 2.9),
        a.set_xlabel("all costs x multiple (1 = central model)"),
        a.set_ylabel("net Sharpe"),
    )
    a.set_title("Net Sharpe against the cost level", loc="left")
    comp = ["cost_spread", "cost_commission", "cost_impact", "cost_borrow"]
    cols = ["#7fb3d5", "#a9cce3", "#b03a2e", "#d5d8dc"]
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    years = 1006 / DAYS_PER_YEAR
    for cmp_, cc in zip(comp, cols, strict=True):
        v = np.array([fr[s][c][cmp_].sum() * 1e4 / years for _, s, c in names])
        b.bar(x, v, bottom=bottom, color=cc, label=cmp_.replace("cost_", ""), width=0.6)
        bottom += v
    b.scatter(
        x,
        [fr[s][c]["pnl"].sum() * 1e4 / years for _, s, c in names],
        marker="D",
        color=PALETTE["ink"],
        zorder=4,
        label="gross P&L",
    )
    b.set_xticks(
        x,
        [n[0].replace("PCA ", "PCA\n").replace("pairs: ", "pairs\n") for n in names],
        fontsize=7.5,
    )
    b.set_ylabel("bps of capital per year"), b.legend(fontsize=7)
    b.set_title("Where the gross goes (costs stack, gross diamond)", loc="left")
    fig.tight_layout()
    P.save(fig, FIG / "08_cost_sensitivity.png")


def fig_capacity(fr: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    caps = np.logspace(5, 10, 60)
    for ax, (title, key, c) in zip(
        axes[:2],
        (
            ("PCA reversal k=5", ("s7", "pca_reversal_k5"), PALETTE["pca_reversal"]),
            ("Pairs: expanding, entry 1.5", ("s6", "expanding_in1.5"), PALETTE["pairs"]),
        ),
        strict=True,
    ):
        d = fr[key[0]][key[1]]
        years = len(d) / DAYS_PER_YEAR
        for m, ls in ((1.0, "-"), (0.5, "--"), (0.25, "-."), (0.0, ":")):
            y = [
                net_at_capital(d, cap, 1e8, spread_mult=m, commission_mult=m, borrow_mult=m).sum()
                * 1e4
                / years
                for cap in caps
            ]
            ax.plot(caps, y, color=c, ls=ls, lw=1.4, label=f"fixed costs x{m:g}")
        ax.axhline(d["pnl"].sum() * 1e4 / years, color=PALETTE["ink"], lw=0.9, label="gross")
        ax.axhline(0, color=PALETTE["bad"], lw=0.9)
        ax.set_yscale("symlog", linthresh=100)
        for m in (1.0, 0.5, 0.25, 0.0):  # where each curve crosses zero
            be = break_even_capital(d, 1e8, spread_mult=m, commission_mult=m, borrow_mult=m)
            if 0 < be < 1e10:
                ax.scatter([be], [0], color=c, edgecolor=PALETTE["ink"], zorder=5, s=28)
        (
            ax.set_xscale("log"),
            ax.set_xlabel("capital ($)"),
            ax.set_ylabel("net return, bps per year"),
        )
        ax.set_title(title, loc="left"), ax.legend(fontsize=7)
    d = fr["s7"]["pca_reversal_k5"]
    for k, ls in (("pca_reversal_k5", "-"), ("pca_sscore_k5", "-"), ("pca_reversal_k15", "--")):
        dd = fr["s7"][k]
        axes[2].plot(
            caps,
            [100 * participation_breach_share(dd["participation"], cap, 1e8) for cap in caps],
            color=P.colour_of(k),
            lw=1.3,
            ls=ls,
            label=k,
        )
    (
        axes[2].set_xscale("log"),
        axes[2].set_xlabel("capital ($)"),
        axes[2].set_ylabel("days with a trade above 10 % of ADV (%)"),
    )
    axes[2].axhline(5, color=PALETTE["ink"], ls="--", lw=0.8), axes[2].legend(fontsize=7)
    axes[2].set_title("Participation (model-based)", loc="left")
    fig.suptitle(
        "Capacity curves are model-based estimates from the cost model, not observed live capacity (dots: break-even capital; symlog axis)",
        y=1.03,
        fontsize=9.5,
        fontweight="bold",
    )
    fig.tight_layout()
    P.save(fig, FIG / "09_capacity_curve.png")


def _label_variant(v: dict) -> str:
    return v["variant"].replace("=", " = ")


def fig_sensitivity(s9: dict) -> None:
    pr = s9["pca"]["pca_reversal_k5"]
    rows = [v for v in pr["variants"] if v["dimension"] != "anchor"]
    anchor = next(v for v in pr["variants"] if v["dimension"] == "anchor")
    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.45, wspace=0.28)
    a = fig.add_subplot(gs[:, 0])
    y = np.arange(len(rows))[::-1]
    a.scatter(
        [r["gross_sharpe"] for r in rows],
        y,
        color=PALETTE["pca_reversal"],
        s=22,
        label="gross",
        zorder=3,
    )
    a.scatter(
        [r["net_sharpe"] for r in rows],
        y,
        facecolors="none",
        edgecolors=PALETTE["bad"],
        s=22,
        label="net",
        zorder=3,
    )
    a.set_yticks(y, [_label_variant(r) for r in rows], fontsize=7)
    a.axvline(0, color=PALETTE["ink"], lw=0.9)
    a.axvline(
        anchor["gross_sharpe"], color=PALETTE["pca_reversal"], ls="--", lw=0.9, label="anchor gross"
    )
    a.set_xlabel("Sharpe ratio"), a.legend(loc="lower left", fontsize=7)
    a.set_title("PCA reversal k=5: one parameter at a time", loc="left")
    g = pd.DataFrame(s9["pca"]["two_way_entry_by_k_reversal"])
    for i, (val, title, cmap) in enumerate(
        (
            ("gross_sharpe", "gross Sharpe: factors x entry threshold", "Blues"),
            ("net_sharpe", "net Sharpe", "Reds_r"),
        )
    ):
        ax = fig.add_subplot(gs[i, 1])
        M = g.pivot(index="n_factors", columns="entry", values=val)
        ax.imshow(M.to_numpy(), cmap=cmap, aspect="auto")
        (
            ax.set_xticks(range(M.shape[1]), [f"{c:g}" for c in M.columns]),
            ax.set_yticks(range(M.shape[0]), [f"k={k}" for k in M.index]),
        )
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                ax.text(c, r, f"{M.iloc[r, c]:+.2f}", ha="center", va="center", fontsize=7.5)
        (
            ax.grid(False),
            ax.set_title(title, loc="left"),
            ax.set_xlabel("entry threshold") if i else None,
        )
    fig.suptitle(
        "Parameter sensitivity: the gross effect is broad, the net result is negative everywhere tried",
        y=0.99,
        fontsize=10,
        fontweight="bold",
    )
    P.save(fig, FIG / "10_sensitivity_pca.png")

    pv = s9["pairs"]["variants"]
    fig, ax = plt.subplots(figsize=(7.5, 8))
    y = np.arange(len(pv))[::-1]
    ax.scatter(
        [v["gross_sharpe"] for v in pv], y, color=PALETTE["pairs"], s=22, label="gross", zorder=3
    )
    ax.scatter(
        [v["net_sharpe"] for v in pv],
        y,
        facecolors="none",
        edgecolors=PALETTE["bad"],
        s=22,
        label="net",
        zorder=3,
    )
    ax.set_yticks(y, [_label_variant(v) for v in pv], fontsize=7)
    (
        ax.axvline(0, color=PALETTE["ink"], lw=0.9),
        ax.set_xlabel("Sharpe ratio"),
        ax.legend(loc="lower left"),
    )
    ax.set_title("Pair strategy: 33 variants (thresholds, windows, hedge, screen)", loc="left")
    P.save(fig, FIG / "11_sensitivity_pairs.png")


def fig_regimes(rb: dict, rf: dict) -> None:
    rt = pd.DataFrame(rb["regime_tests"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    order = [
        "pca_reversal_k5",
        "pca_reversal_k10",
        "pca_reversal_k15",
        "pca_sscore_k5",
        "pca_sscore_k10",
        "pca_sscore_k15",
    ]
    d = rt[(rt.regime == "disp_high")].set_index("strategy")
    x = np.arange(len(order))
    axes[0].bar(
        x - 0.19,
        [d.loc[c, "sharpe_true"] for c in order],
        0.36,
        color=PALETTE["pca_reversal"],
        label="high dispersion",
    )
    axes[0].bar(
        x + 0.19,
        [d.loc[c, "sharpe_false"] for c in order],
        0.36,
        color=PALETTE["placebo"],
        label="low dispersion",
    )
    axes[0].axhline(0, color=PALETTE["ink"], lw=0.9)
    (
        axes[0].set_xticks(
            x, [c.replace("pca_", "").replace("_", "\n") for c in order], fontsize=7.5
        ),
        axes[0].legend(),
    )
    (
        axes[0].set_ylabel("gross Sharpe"),
        axes[0].set_title("PCA strategies by dispersion regime", loc="left"),
    )
    regs = [("vol_high", "volatility"), ("trend_up", "trend"), ("disp_high", "dispersion")]
    strategies = list(rt.strategy.unique())
    for k, (rg, _) in enumerate(regs):
        sub = rt[rt.regime == rg].set_index("strategy")
        xs = np.full(len(strategies), k) + np.linspace(-0.3, 0.3, len(strategies))
        sig = np.array([sub.loc[s, "p_bh"] < 0.10 for s in strategies])
        cols = [P.colour_of(s) for s in strategies]
        axes[1].scatter(
            xs[~sig],
            [sub.loc[s, "difference"] for s, m in zip(strategies, sig, strict=True) if not m],
            c=[c for c, m in zip(cols, sig, strict=True) if not m],
            s=22,
            alpha=0.9,
        )
        axes[1].scatter(
            xs[sig],
            [sub.loc[s, "difference"] for s, m in zip(strategies, sig, strict=True) if m],
            c=[c for c, m in zip(cols, sig, strict=True) if m],
            s=60,
            edgecolors=PALETTE["ink"],
            linewidths=1.2,
        )
    axes[1].axhline(0, color=PALETTE["ink"], lw=0.9)
    (
        axes[1].set_xticks(range(3), [n for _, n in regs]),
        axes[1].set_ylabel("Sharpe(first state) - Sharpe(second state)"),
    )
    axes[1].set_title("All 45 regime tests (ringed: BH q<0.10)", loc="left")
    from matplotlib.lines import Line2D

    axes[1].legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=PALETTE[k], label=v)
            for k, v in (
                ("pca_reversal", "PCA reversal"),
                ("pca_sscore", "PCA s-score"),
                ("pairs", "pairs"),
            )
        ],
        loc="lower left",
        fontsize=7,
    )
    by = rf["by_year_pca"]["pca_reversal_k5"]
    yrs = list(by)
    x = np.arange(len(yrs))
    axes[2].bar(
        x - 0.19,
        [by[y]["sharpe_high"] for y in yrs],
        0.36,
        color=PALETTE["pca_reversal"],
        label="high dispersion",
    )
    axes[2].bar(
        x + 0.19,
        [by[y]["sharpe_low"] for y in yrs],
        0.36,
        color=PALETTE["placebo"],
        label="low dispersion",
    )
    (
        axes[2].axhline(0, color=PALETTE["ink"], lw=0.9),
        axes[2].set_xticks(x, yrs),
        axes[2].legend(loc="upper left", ncol=2),
    )
    axes[2].set_ylim(top=8)
    axes[2].set_title("Reversal k=5: same sign in every year (post-hoc check)", loc="left")
    fig.tight_layout()
    P.save(fig, FIG / "12_regimes.png")


# ---- statistical validation ---------------------------------------------------------------------
def fig_forest(fr: dict, s8: dict) -> None:
    labels = list(s8["gross"])
    rows = []
    for lab in labels:
        d = (fr["s7"] if lab.startswith("pca") else fr["s5"])[lab]["pnl"].to_numpy()
        est, (lo, hi), _ = bootstrap_ci(d, metrics.sharpe, n_boot=2000, mean_block=10, seed=1)
        rows.append((lab, est, lo, hi))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for i, (lab, e, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=P.colour_of(lab), lw=2, alpha=0.8)
        ax.scatter([e], [i], color=P.colour_of(lab), s=26, zorder=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    ax.axvline(0, color=PALETTE["ink"], lw=0.9)
    for n, ls, lab in (
        (24, "--", "luck benchmark SR0, N = 24"),
        (s8["effective_trials_gross"]["n_eff_average_correlation"], ":", "SR0, N_eff = 9.3"),
    ):
        sr0 = expected_max_sharpe(n, s8["var_daily_sharpe"]["gross"]) * np.sqrt(DAYS_PER_YEAR)
        ax.axvline(sr0, color=PALETTE["bad"], ls=ls, lw=1.1, label=f"{lab}: {sr0:.2f}")
    (
        ax.set_xlabel("gross Sharpe ratio, 95 % stationary-bootstrap interval"),
        ax.legend(loc="lower right", fontsize=7.5),
    )
    ax.set_title(
        "Only reversal k=5 excludes zero, and it sits just above the selection benchmark",
        loc="left",
        fontsize=9,
    )
    P.save(fig, FIG / "13_sharpe_forest.png")


def fig_deflated(s8: dict, s9: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    years = s8["selection_luck_table"]["sample_years"]
    n_days = s8["n_days"]
    Ns = np.unique(np.round(np.logspace(0, 3, 60)).astype(int))
    best = s8["gross"]["pca_reversal_k5"]
    axes[0].plot(
        Ns,
        [expected_max_sharpe(n, 1.0 / (n_days - 1)) * np.sqrt(DAYS_PER_YEAR) for n in Ns],
        color=PALETTE["bad"],
        lw=1.6,
        label=f"best of N luck-only strategies ({years:.1f} years)",
    )
    axes[0].axhline(
        best["sharpe_annual"],
        color=PALETTE["pca_reversal"],
        lw=1.3,
        label=f"observed best gross: {best['sharpe_annual']:.2f}",
    )
    (
        axes[0].axvline(24, color=PALETTE["ink"], ls="--", lw=0.8),
        axes[0].text(25, 0.05, "N = 24", fontsize=7.5),
    )
    (
        axes[0].set_xscale("log"),
        axes[0].set_xlabel("number of strategies tried, N"),
        axes[0].set_ylabel("annualised Sharpe"),
        axes[0].legend(fontsize=7),
    )
    axes[0].set_title("Selection luck", loc="left")
    eff = s8["effective_trials_gross"]
    n_surface = 24 + s9["surface_as_trials"]["n_configurations_evaluated"]
    cases = [
        ("no deflation\n(N=1)", best["psr_vs_zero"]),
        (f"N_eff={eff['n_eff_participation_ratio']:.1f}", best["dsr_neff_part"]),
        (f"N_eff={eff['n_eff_average_correlation']:.1f}", best["dsr_neff_corr"]),
        ("N=15", best["dsr_n15"]),
        ("N=24\n(registry)", best["dsr_n24"]),
        (f"N={n_surface}\n(+ surface)", s9["surface_as_trials"]["dsr_if_n_is_24_plus_surface"]),
    ]
    axes[1].bar(
        range(len(cases)),
        [c[1] for c in cases],
        color=[PALETTE["muted"]] + [PALETTE["pca_reversal"]] * 5,
        width=0.65,
    )
    (
        axes[1].axhline(0.95, color=PALETTE["bad"], ls="--", lw=1.1),
        axes[1].text(5.45, 0.965, "0.95", color=PALETTE["bad"], fontsize=8, ha="right"),
    )
    (
        axes[1].set_xticks(range(len(cases)), [c[0] for c in cases], fontsize=7),
        axes[1].set_ylim(0, 1.05),
    )
    (
        axes[1].set_ylabel("probability true Sharpe > luck benchmark"),
        axes[1].set_title("Deflated Sharpe, PCA reversal k=5", loc="left"),
    )
    for i, c in enumerate(cases):
        axes[1].text(i, c[1] + 0.015, f"{c[1]:.2f}", ha="center", fontsize=7.5)
    axes[2].plot(Ns, [min_backtest_years(n, 1.0) for n in Ns], color=PALETTE["ink"], lw=1.5)
    axes[2].axhline(
        years,
        color=PALETTE["pca_reversal"],
        ls="--",
        lw=1.1,
        label=f"data available: {years:.1f} years",
    )
    (
        axes[2].set_xscale("log"),
        axes[2].set_xlabel("N"),
        axes[2].set_ylabel("years"),
        axes[2].legend(fontsize=7),
    )
    axes[2].set_title(
        "Backtest length needed before a Sharpe of 1 is not luck", loc="left", fontsize=8.5
    )
    fig.tight_layout()
    P.save(fig, FIG / "14_deflated_sharpe.png")


def fig_placebo(pipe, s7: dict, fr: dict) -> None:
    from statarb.backtest.costs import CostConfig
    from statarb.backtest.pca_walkforward import (
        PCAStrategyConfig,
        fold_signals,
        placebo_permutation,
        prepare_pca_fold,
        run_fold,
    )
    from statarb.backtest.walkforward import make_folds
    from statarb.selection.screening import ScreenConfig
    from statarb.signals.residuals import ResidualConfig

    seed, n_draws = s7["seed"], s7["n_draws"]
    folds = make_folds(pipe.cfg.split, pipe.calendar, 2015, 2018, 4, "rolling")
    cfg = PCAStrategyConfig(residual=ResidualConfig(n_factors=5, signal="reversal"))
    fss = [
        fold_signals(prepare_pca_fold(pipe, f, ScreenConfig()), cfg, CostConfig(capital=1e8))
        for f in folds
    ]
    sharpes = []
    for d in range(n_draws):
        parts = [
            run_fold(
                fs,
                cfg,
                None,
                placebo_permutation(fs, np.random.default_rng([seed, fs.fd.fold.index, d])),
            )
            for fs in fss
        ]
        sharpes.append(metrics.sharpe(pd.concat(parts)["pnl"].to_numpy()))
    sharpes = np.array(sharpes)
    saved = s7["results"]["pca_reversal_k5"]["placebo"]["gross_sharpe_median"]
    assert abs(np.median(sharpes) - saved) < 1e-9, (
        f"placebo not reproduced: {np.median(sharpes)} vs {saved}"
    )
    real = fr["s7"]["pca_reversal_k5"]["pnl"].to_numpy()
    est, (lo, hi), boot = bootstrap_ci(real, metrics.sharpe, n_boot=2000, mean_block=10, seed=1)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    bins = np.linspace(-2, 2.6, 60)
    ax.hist(
        sharpes,
        bins=bins,
        color=PALETTE["placebo"],
        alpha=0.9,
        density=True,
        label=f"placebo: relabelled books ({n_draws} draws)",
    )
    ax.hist(
        boot,
        bins=bins,
        color=PALETTE["pca_reversal"],
        alpha=0.55,
        density=True,
        label="real book: bootstrap of its own Sharpe",
    )
    (
        ax.axvline(est, color=PALETTE["pca_reversal"], lw=1.6),
        ax.axvline(
            np.quantile(sharpes, 0.95),
            color=PALETTE["ink"],
            ls="--",
            lw=1.1,
            label="placebo 95th percentile",
        ),
    )
    (
        ax.set_xlabel("gross Sharpe ratio"),
        ax.set_ylabel("density"),
        ax.legend(fontsize=7.5, loc="upper left"),
    )
    ax.set_title(
        f"PCA reversal k=5: {est:.2f} = placebo percentile {100 * np.mean(sharpes < est):.1f}; 95 % CI [{lo:.2f}, {hi:.2f}]",
        loc="left",
        fontsize=9,
    )
    P.save(fig, FIG / "15_bootstrap_vs_placebo.png")


def fig_multiple_testing(s8: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    fams = [("all_15", "all 15"), ("pairs_9", "9 pairs"), ("pca_6", "6 PCA")]
    fg = s8["family_gross"]
    x = np.arange(3)
    axes[0].bar(
        x - 0.27,
        [fg[k]["reality_check_p"] for k, _ in fams],
        0.26,
        color=PALETTE["pca_sscore"],
        label="Reality Check",
    )
    axes[0].bar(
        x,
        [fg[k]["spa_p"]["consistent"] for k, _ in fams],
        0.26,
        color=PALETTE["pca_reversal"],
        label="SPA (consistent)",
    )
    axes[0].bar(
        x + 0.27,
        [min(v["p_romano_wolf"] for v in fg[k]["per_strategy"].values()) for k, _ in fams],
        0.26,
        color=PALETTE["pairs"],
        label="Romano-Wolf, best strategy",
    )
    (
        axes[0].axhline(0.05, color=PALETTE["bad"], ls="--", lw=1),
        axes[0].set_xticks(x, [n for _, n in fams]),
        axes[0].set_ylabel("p-value (best of the family)"),
        axes[0].legend(fontsize=7, loc="upper center"),
        axes[0].set_ylim(0, 0.66),
    )
    axes[0].set_title("Family-wise tests (gross)", loc="left")
    axes[1].bar(
        x,
        [s8["pbo_gross"][k]["blocks16"]["pbo"] for k, _ in fams],
        color=PALETTE["muted"],
        width=0.55,
    )
    (
        axes[1].axhline(0.5, color=PALETTE["ink"], ls="--", lw=1),
        axes[1].set_xticks(x, [n for _, n in fams]),
        axes[1].set_ylim(0, 1),
    )
    (
        axes[1].set_ylabel("probability of backtest overfitting"),
        axes[1].set_title(
            "CSCV PBO (0.5 = selection carries no information)", loc="left", fontsize=8.5
        ),
    )
    sc = s8["screen_fdr"]
    names = list(sc)
    xs = np.arange(len(names))
    axes[2].bar(
        xs - 0.2,
        [sc[n]["calibrated_discoveries_at_5pct_uncorrected"] for n in names],
        0.38,
        color=PALETTE["pairs"],
        label="calibrated p <= 0.05",
    )
    axes[2].bar(
        xs + 0.2,
        [sc[n]["expected_under_null_at_5pct"] for n in names],
        0.38,
        color=PALETTE["placebo"],
        label="expected if all null",
    )
    (
        axes[2].set_xticks(
            xs,
            [n.replace("stage3_2011_2015", "Stage 3").replace("fold_test_", "") for n in names],
            fontsize=8,
        ),
        axes[2].legend(fontsize=7),
    )
    (
        axes[2].set_ylim(0, 88),
        axes[2].set_ylabel("pairs"),
        axes[2].set_title(
            "Pair screens: discoveries vs chance (BH: 0 at q=5 %)", loc="left", fontsize=8.5
        ),
    )
    fig.tight_layout()
    P.save(fig, FIG / "16_multiple_testing.png")


def fig_stability(rb: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    fs = rb["factor_space"]
    lags = [1, 3, 6, 12, 24, 36, 60]
    for k, col in ((5, PALETTE["pca_reversal"]), (10, PALETTE["pca_sscore"])):
        a.plot(
            lags,
            [fs[f"k{k}"]["overlap_by_lag_months"][str(lag)] for lag in lags],
            marker="o",
            color=col,
            label=f"leading {k}-factor space",
        )
        a.axhline(fs[f"k{k}"]["random_level_k_over_N"], color=col, ls=":", lw=0.9)
    a.plot(
        lags,
        [fs["first_eigenvector_abs_cosine_by_lag_months"][str(lag)] for lag in lags],
        marker="s",
        color=PALETTE["ink"],
        label="market eigenvector (|cos|)",
    )
    a.set_xscale("log"), a.set_xticks(lags, [str(lag) for lag in lags]), a.set_ylim(0, 1.03)
    (
        a.set_xlabel("months between fits"),
        a.set_ylabel("overlap with the earlier fit"),
        a.legend(fontsize=7.5, loc="lower left"),
    )
    a.set_title(
        "The PCA factor space drifts slowly (dotted: random subspaces)", loc="left", fontsize=8.5
    )
    bt = pd.DataFrame(rb["break_tests"]).sort_values("p_value")
    y = np.arange(len(bt))[::-1]
    b.scatter(bt.p_value, y, c=[P.colour_of(s) for s in bt.strategy], s=24, zorder=3)
    b.axvline(0.05, color=PALETTE["bad"], ls="--", lw=1)
    (
        b.set_yticks(y, bt.strategy, fontsize=7),
        b.set_xlabel("sup-F mean-break p-value"),
        b.set_xlim(0, 1),
    )
    b.set_title("No break in any strategy's mean (low power)", loc="left")
    fig.tight_layout()
    P.save(fig, FIG / "17_stability.png")


def main() -> None:
    warnings.filterwarnings("ignore")
    P.apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    fr = load_frames()
    s7, s8, s9, rb, rf = (
        J("stage7_pca.json"),
        J("stage8_multiple_testing.json"),
        J("stage9_sensitivity.json"),
        J("stage9_regimes_breaks.json"),
        J("stage9_regime_followup.json"),
    )
    sel = pd.read_csv(R / "stage3_selected_pairs_2011_2015.csv")
    for f, args in (
        (fig_pair_screen, (s8,)),
        (fig_oos_persistence, (sel, s9)),
        (fig_equity, (fr,)),
        (fig_risk, (fr,)),
        (fig_turnover_exposure, (fr,)),
        (fig_cost, (fr,)),
        (fig_capacity, (fr,)),
        (fig_sensitivity, (s9,)),
        (fig_regimes, (rb, rf)),
        (fig_forest, (fr, s8)),
        (fig_deflated, (s8, s9)),
        (fig_multiple_testing, (s8,)),
        (fig_stability, (rb,)),
    ):
        f(*args)
        print("wrote", f.__name__, flush=True)
    from statarb.config import load_config
    from statarb.data.pipeline import DataPipeline

    pipe = DataPipeline(load_config("configs/data_default.yaml"))
    assert pipe.cfg.split.phase_of(pd.Timestamp("2016-12-30")) == "research"
    fig_pairs(
        pipe,
        sel,
        [
            ("CRM", "ADBE", "Rank 1 in training (most significant)", ""),
            ("EL", "KMB", "Best out-of-sample persistence of the 20", ""),
        ],
        "02_pair_examples.png",
        "Selected pairs, formed on 2011-2015 and traded in 2016 (frozen hedge ratio): a success in training does not carry over",
    )
    fig_pairs(
        pipe,
        sel,
        [
            (
                "DHR",
                "HSIC",
                "Data-event failure: DHR spin-off booked as a ~36 % dividend\n(dash-dot: ex-date; the walk-forward engine blacks these out)",
                "2016-07-05",
            ),
            ("AMAT", "MU", "Genuine failure: the spread mean shifts by 7.6 sd", ""),
        ],
        "03_pair_failures.png",
        "Representative failures: a corporate-action artefact (left) and a genuine break in the relationship (right)",
    )
    fig_placebo(pipe, s7, fr)
    print("wrote fig_placebo")
    pipe.close()


if __name__ == "__main__":
    sys.exit(main())
