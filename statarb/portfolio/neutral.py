"""Turn position states into a dollar- and factor-neutral book.

Sizing: every held name gets ``name_size`` of capital scaled by ``median(sigma_E) / sigma_E``
(inverse residual volatility, so each name contributes similar residual risk), clipped at
``max_mult``.  A name's *gross* is therefore fixed, like the pair strategy's fixed slot: the number
of open names, not a rescaling, sets the book's gross exposure, and days with few signals are simply
under-invested.

Neutralisation: the smallest change (in the Euclidean norm of the weights) that makes the book
exactly neutral to the ``k`` fitted factors and to dollars::

    w' = w - G (G'G)^-1 G' w,     G = [1 | B]      (N x (k+1), B = the stocks' factor betas)

so that ``sum(w') = 0`` and ``B'w' = 0`` to machine precision.  The hedge is spread across the whole
tradable universe (it is what trading the eigenportfolios themselves amounts to), it can flip the
sign of a very small position, and it changes daily -- its turnover is charged like any other.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class BookConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry: float = Field(1.25, gt=0)
    exit: float = Field(0.5, ge=0)
    stop: float = Field(4.0, gt=0)
    max_hold: int = Field(60, ge=1)
    name_size: float = Field(0.02, gt=0, description="capital fraction per held name at median vol")
    max_mult: float = Field(2.5, gt=0)


def neutralise(w: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """Project ``w`` (N,) onto the null space of ``[1 | exposures]`` (``exposures``: N x m, m >= 0).

    With ``m = 0`` (the no-factor control) the book is made dollar-neutral only.
    """
    g = np.column_stack([np.ones(len(w)), exposures])
    coef = np.linalg.lstsq(g, w, rcond=None)[0]
    return w - g @ coef


def size_weights(state: np.ndarray, sigma_e: np.ndarray, cfg: BookConfig) -> np.ndarray:
    """Raw (not yet neutral) weights (D, N): ``state * name_size * clip(median sigma / sigma)``."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.nanmedian(sigma_e, axis=1, keepdims=True)
        mult = np.clip(ref / sigma_e, 0.0, cfg.max_mult)
    return np.where(state != 0, state * cfg.name_size * np.nan_to_num(mult), 0.0)


def neutral_book(
    state: np.ndarray, sigma_e: np.ndarray, beta: np.ndarray, tradable: np.ndarray, cfg: BookConfig
) -> np.ndarray:
    """(D, N) neutral weights.  Days with no position are flat (no hedge is put on for nothing)."""
    raw = size_weights(np.where(tradable, state, 0), sigma_e, cfg)
    out = np.zeros_like(raw)
    usable = tradable & np.isfinite(sigma_e) & np.isfinite(beta).all(axis=2)
    for d in np.flatnonzero(np.any(raw != 0.0, axis=1)):
        u = usable[d]
        w = np.zeros(raw.shape[1])
        w[u] = neutralise(raw[d, u], beta[d, u])
        out[d] = w
    return out
