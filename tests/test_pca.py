"""PCA factor model: agreement with reference implementations, algebraic identities, recovery of
planted structure, and the fitted-scale contract (new data is never re-standardised)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from statarb.models.pca import (
    PCAFit,
    fit_pca,
    marchenko_pastur_upper,
    n_factors_marchenko_pastur,
)


def planted(t=900, n=60, k=3, seed=0, idio=0.012, strength=(0.010, 0.006, 0.004)):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, (t, k)) * np.array(strength[:k])
    b = rng.normal(0, 1, (n, k))
    b[:, 0] = np.abs(b[:, 0]) + 0.5  # a "market" factor: same-signed loadings
    return f @ b.T + rng.normal(0, idio, (t, n)), b


def test_eigenvalues_match_the_correlation_matrix_and_sklearn():
    x, _ = planted()
    fit = fit_pca(x)
    corr = np.corrcoef(x, rowvar=False)
    assert np.allclose(fit.eigenvalues, np.sort(np.linalg.eigvalsh(corr))[::-1], atol=1e-10)
    z = (x - x.mean(0)) / x.std(0, ddof=1)
    sk = PCA(n_components=6, svd_solver="full").fit(z)
    assert np.allclose(fit.eigenvalues[:6], sk.explained_variance_, atol=1e-9)
    for j in range(6):  # eigenvectors agree up to sign
        assert abs(fit.components[:, j] @ sk.components_[j]) == pytest.approx(1.0, abs=1e-8)


def test_eigenvalues_sum_to_n_and_are_descending_and_components_orthonormal():
    x, _ = planted(n=40)
    fit = fit_pca(x)
    assert fit.eigenvalues.sum() == pytest.approx(40, rel=1e-10)
    assert np.all(np.diff(fit.eigenvalues) <= 1e-12)
    assert np.allclose(fit.components.T @ fit.components, np.eye(40), atol=1e-10)
    assert fit.explained_ratio(40) == pytest.approx(1.0)
    assert fit.explained_ratio(3) == pytest.approx(fit.eigenvalues[:3].sum() / 40)


def test_decomposition_identities():
    x, _ = planted(t=400, n=30)
    fit = fit_pca(x)
    k = 4
    z = fit.standardise(x)
    f = fit.factor_returns(x, k)
    eps = fit.residuals(x, k)
    assert np.allclose(f @ fit.loadings(k).T + eps, z, atol=1e-10)  # Z = F V' + eps
    assert np.allclose(f.T @ eps, 0.0, atol=1e-8)  # residual orthogonal to the factors
    assert np.allclose(f.var(axis=0, ddof=1), fit.eigenvalues[:k], atol=1e-10)
    assert np.allclose(np.cov(f, rowvar=False) - np.diag(fit.eigenvalues[:k]), 0, atol=1e-10)
    assert np.allclose(fit.residuals(x, 30), 0.0, atol=1e-9)  # all components: nothing left
    assert np.allclose(fit.residuals(x, 0), z)  # no components: nothing removed


def test_planted_factors_are_recovered_and_marchenko_pastur_counts_them():
    x, b = planted(t=1500, n=60, k=3, seed=4)
    fit = fit_pca(x)
    assert fit.marchenko_pastur_k() == 3
    assert n_factors_marchenko_pastur(fit.eigenvalues, 1500) == 3
    # the fitted 3-d space contains the planted loadings (standardised): tiny out-of-span part
    bz = b / x.std(0, ddof=1)[:, None]
    v = fit.components[:, :3]
    out_of_span = bz - v @ (v.T @ bz)
    assert np.linalg.norm(out_of_span) / np.linalg.norm(bz) < 0.12
    assert fit.explained_ratio(3) > 0.5 > fit.explained_ratio(0)


def test_pure_noise_has_almost_no_eigenvalue_above_the_marchenko_pastur_edge():
    rng = np.random.default_rng(1)
    counts = [fit_pca(rng.normal(size=(600, 100))).marchenko_pastur_k() for _ in range(20)]
    assert np.mean(counts) < 0.5  # the edge is the noise maximum; finite-sample overshoot is rare
    assert marchenko_pastur_upper(400, 100) == pytest.approx(2.25)
    assert marchenko_pastur_upper(100, 100) == pytest.approx(4.0)


def test_new_data_is_standardised_with_the_fitted_mean_and_scale_not_its_own():
    x, _ = planted(t=300, n=20)
    fit = fit_pca(x)
    shifted = x[:50] + 0.5
    assert np.allclose(fit.standardise(shifted), (shifted - fit.mean) / fit.scale)
    assert not np.allclose(fit.standardise(shifted).mean(0), 0.0, atol=1e-3)
    # scaling the new data changes the standardised values (nothing is re-estimated on it)
    assert np.allclose(fit.standardise(2 * x[:10]), (2 * x[:10] - fit.mean) / fit.scale)


def test_sign_convention_and_reordering_equivariance():
    x, _ = planted(t=300, n=25)
    fit = fit_pca(x)
    top = np.argmax(np.abs(fit.components), axis=0)
    assert np.all(fit.components[top, np.arange(25)] > 0)
    perm = np.random.default_rng(2).permutation(25)
    fit2 = fit_pca(x[:, perm])
    assert np.allclose(fit2.eigenvalues, fit.eigenvalues, atol=1e-9)
    for j in range(5):  # first eigenvectors are simple: same up to the reordering
        assert np.allclose(fit2.components[:, j], fit.components[perm, j], atol=1e-7)


def test_dataframe_input_keeps_tickers_and_bad_input_is_rejected():
    x, _ = planted(t=200, n=10)
    df = pd.DataFrame(x, columns=list("abcdefghij"))
    assert fit_pca(df).tickers == list("abcdefghij")
    assert isinstance(fit_pca(df), PCAFit)
    bad = x.copy()
    bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_pca(bad)
    with pytest.raises(ValueError, match="too few"):
        fit_pca(x[:8])
    with pytest.raises(ValueError, match="too few"):
        fit_pca(rng_wide := np.random.default_rng(0).normal(size=(50, 80)))
    assert rng_wide.shape == (50, 80)
    flat = x.copy()
    flat[:, 4] = 1.0
    with pytest.raises(ValueError, match="zero variance"):
        fit_pca(flat)
