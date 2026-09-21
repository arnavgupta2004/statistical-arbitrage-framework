"""Family-wise and false-discovery-rate control for a search over many strategies or pairs.

All bootstrap procedures resample **time** with the stationary bootstrap and apply the *same*
resampled dates to every strategy, so the cross-sectional dependence between strategies is preserved
(that dependence is what makes the maximum of ``K`` strategies less extreme than ``K`` independent
ones).  ``d`` is a (T, K) matrix of performance *relative to a benchmark* (here: daily returns,
benchmark zero).  The null is that **no** strategy has a positive expected performance.

* ``reality_check``  White (2000): ``max_k sqrt(n) dbar_k`` against its bootstrap distribution with
  every strategy recentred at zero (the least favourable null).  Conservative when many strategies
  are clearly poor, because they still add noise to the maximum.
* ``spa_test``  Hansen (2005): studentised, floored at zero, and poor strategies are recentred at
  their own (negative) mean so they stop diluting the test.  Three p-values: ``lower`` (recentre
  every negative mean; liberal), ``consistent`` (recentre only clearly poor strategies; the
  recommended one) and ``upper`` (recentre nothing).  ``lower <= consistent <= upper``.
* ``romano_wolf``  step-down procedure: family-wise-adjusted p-values for *each* strategy, i.e.
  which ones survive when the best of the family is being reported (Romano & Wolf 2005).
* ``benjamini_hochberg`` / ``benjamini_yekutieli``  false-discovery-rate control on a vector of
  p-values (BH: independence or positive dependence; BY: any dependence, at a log(m) price).
* ``cscv_pbo``  Bailey, Borwein, Lopez de Prado & Zhu (2015): probability of backtest overfitting.
  The time axis is cut into ``S`` blocks; over every way of taking half of them as "in-sample", the
  in-sample-best strategy is ranked among all strategies on the remaining half.  ``PBO`` is the
  share of splits where it lands at or below the median out of sample: ~0.5 means the selection
  carries no information, near 0 that the in-sample winner reliably stays a winner.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats

from statarb.statistics.bootstrap import stationary_bootstrap_matrix


# ---- false discovery rate ---------------------------------------------------------------------
def _step_up(p: np.ndarray, q: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    m = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m * scale / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj <= q, adj


def benjamini_hochberg(p, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Rejection mask and BH-adjusted p-values (a p-value is rejected iff adjusted <= q)."""
    p = np.asarray(p, dtype=float)
    return _step_up(p, q, 1.0)


def benjamini_yekutieli(p, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    return _step_up(p, q, float(np.sum(1.0 / np.arange(1, len(p) + 1))))


# ---- bootstrap tests over a family of strategies ----------------------------------------------
def _boot_means(d: np.ndarray, n_boot: int, mean_block: float, seed: int, chunk: int = 200):
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, d.shape[1]))
    for lo in range(0, n_boot, chunk):
        hi = min(lo + chunk, n_boot)
        idx = stationary_bootstrap_matrix(n, hi - lo, mean_block, rng)
        out[lo:hi] = d[idx].mean(axis=1)
    return out


def _prepare(d, n_boot, mean_block, seed):
    d = np.asarray(d, dtype=float)
    n, k = d.shape
    if not np.all(np.isfinite(d)):
        raise ValueError("d must be finite")
    dbar = d.mean(axis=0)
    boot = _boot_means(d, n_boot, mean_block, seed)
    omega = np.sqrt(n) * boot.std(axis=0, ddof=1)  # bootstrap sd of sqrt(n) * mean
    return n, k, dbar, boot, omega


def _p(stat_boot: np.ndarray, stat: float) -> float:
    return float((1 + np.sum(stat_boot >= stat)) / (len(stat_boot) + 1))


def reality_check(d, n_boot: int = 2000, mean_block: float = 10.0, seed: int = 0) -> dict:
    n, _, dbar, boot, _ = _prepare(d, n_boot, mean_block, seed)
    stat = float(np.sqrt(n) * dbar.max())
    stat_b = np.sqrt(n) * (boot - dbar).max(axis=1)
    return {"statistic": stat, "p_value": _p(stat_b, stat), "best": int(np.argmax(dbar))}


def spa_test(d, n_boot: int = 2000, mean_block: float = 10.0, seed: int = 0) -> dict:
    n, _, dbar, boot, omega = _prepare(d, n_boot, mean_block, seed)
    ok = omega > 0
    t = np.where(ok, np.sqrt(n) * dbar / np.where(ok, omega, 1.0), 0.0)
    stat = float(max(t.max(), 0.0))
    poor = t <= -np.sqrt(2.0 * np.log(np.log(n)))
    mus = {
        "lower": np.minimum(dbar, 0.0),
        "consistent": np.where(poor, dbar, 0.0),
        "upper": np.zeros_like(dbar),
    }
    out = {"statistic": stat, "best": int(np.argmax(t)), "n_poor": int(poor.sum())}
    for name, mu in mus.items():
        z = np.where(ok, np.sqrt(n) * (boot - dbar + mu) / np.where(ok, omega, 1.0), 0.0)
        out[name] = _p(np.maximum(z.max(axis=1), 0.0), stat)
    return out


def romano_wolf(d, n_boot: int = 2000, mean_block: float = 10.0, seed: int = 0) -> dict:
    """Raw one-sided bootstrap p-values and step-down family-wise adjusted p-values."""
    n, k, dbar, boot, omega = _prepare(d, n_boot, mean_block, seed)
    ok = omega > 0
    scale = np.where(ok, omega, np.inf)
    t = np.sqrt(n) * dbar / scale
    tb = np.sqrt(n) * (boot - dbar) / scale
    raw = np.array([_p(tb[:, j], t[j]) for j in range(k)])
    order = np.argsort(-t, kind="stable")
    adj = np.empty(k)
    running = 0.0
    for rank, j in enumerate(order):
        remaining = order[rank:]
        p_j = _p(tb[:, remaining].max(axis=1), t[j])
        running = max(running, p_j)
        adj[j] = running
    return {"t": t, "p_raw": raw, "p_adjusted": adj, "order": order}


# ---- probability of backtest overfitting ------------------------------------------------------
def _block_sharpe(sums, sq, cnt):
    mean = sums / cnt
    var = (sq - cnt * mean**2) / (cnt - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(var > 0, mean / np.sqrt(var), np.nan)


def cscv_pbo(returns, n_blocks: int = 16) -> dict:
    """Probability of backtest overfitting over all C(S, S/2) combinatorial splits."""
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    if n_blocks % 2 or n_blocks < 4 or t < n_blocks * 4 or n < 2:
        raise ValueError("need an even n_blocks >= 4, several rows per block, and >= 2 trials")
    edges = np.linspace(0, t, n_blocks + 1).astype(int)
    sums = np.array([x[a:b].sum(0) for a, b in zip(edges[:-1], edges[1:], strict=True)])
    sq = np.array([(x[a:b] ** 2).sum(0) for a, b in zip(edges[:-1], edges[1:], strict=True)])
    cnt = np.diff(edges).astype(float)
    logits, is_sr, oos_sr = [], [], []
    for ins in combinations(range(n_blocks), n_blocks // 2):
        ins = list(ins)
        oos = [b for b in range(n_blocks) if b not in ins]
        s_in = _block_sharpe(sums[ins].sum(0), sq[ins].sum(0), cnt[ins].sum())
        s_out = _block_sharpe(sums[oos].sum(0), sq[oos].sum(0), cnt[oos].sum())
        best = int(np.nanargmax(s_in))
        rank = stats.rankdata(np.nan_to_num(s_out, nan=-np.inf))[best]  # 1 = worst
        omega = rank / (n + 1)
        logits.append(np.log(omega / (1.0 - omega)))
        is_sr.append(s_in[best])
        oos_sr.append(s_out[best])
    logits, is_sr, oos_sr = np.array(logits), np.array(is_sr), np.array(oos_sr)
    slope = np.polyfit(is_sr, oos_sr, 1)[0] if np.std(is_sr) > 0 else float("nan")
    return {
        "pbo": float(np.mean(logits <= 0)),
        "n_splits": len(logits),
        "mean_logit": float(logits.mean()),
        "is_oos_slope": float(slope),
        "share_oos_sharpe_negative": float(np.mean(oos_sr < 0)),
        "mean_is_best_sharpe": float(is_sr.mean()),
        "mean_oos_sharpe_of_is_best": float(oos_sr.mean()),
    }
