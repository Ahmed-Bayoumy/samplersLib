"""
Validation of the reduced-space TPE path (`BiTPE` with `n_reduced_dims < D`)
and of `Reducers`: the properties the method description relies on.
"""

import numpy as np
import pytest

from samplersLib.reducers import Reducers
from samplersLib.samplers import BiTPE


# pylint: disable=missing-function-docstring
def make_split(n=40, dim=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, dim))
    F = np.sum((X - 0.3) ** 2, axis=1)
    order = np.argsort(F)
    g, b = order[: n // 4], order[n // 4 :]
    return X[g], F[g], X[b], F[b], np.array([[0.0, 1.0]] * dim)


def make_tpe(seed=7, lift="mean", method="pod", k=3, n_r=32, **data_kw):
    gX, gF, bX, bF, vlim = make_split(**data_kw)
    return BiTPE(
        good_data=gX,
        good_f_values=gF,
        bad_data=bX,
        bad_f_values=bF,
        kernel_type={"Gaussian": 1.0},
        n_r=n_r,
        vlim=vlim,
        seed=seed,
        n_reduced_dims=k,
        reduction_method=method,
        lift=lift,
    )


@pytest.mark.parametrize("method", ["pca", "pod", "random"])
def test_lift_stays_inside_bounds(method):
    rng = np.random.default_rng(1)
    vlim = np.column_stack([-np.arange(1, 9), np.arange(1, 9) * 2.0])
    X = rng.uniform(vlim[:, 0], vlim[:, 1], size=(30, 8))
    red = Reducers(data=X, vlim=vlim, nd=8, n_components=3, method=method)
    red.rd()
    far = rng.normal(scale=50.0, size=(100, 3))
    lifted = red.project_rd_to_original_space(far)
    assert np.all(lifted >= vlim[:, 0]) and np.all(lifted <= vlim[:, 1])


@pytest.mark.parametrize("method", ["pca", "pod"])
def test_exact_reconstruction_of_rank_k_data(method):
    # Points on a 3-dimensional affine subspace are recovered exactly by
    # reduce -> lift when k = 3.
    rng = np.random.default_rng(2)
    D, k = 10, 3
    basis, _ = np.linalg.qr(rng.standard_normal((D, k)))
    X = 0.5 + 0.1 * rng.uniform(-1, 1, size=(25, k)) @ basis.T
    vlim = np.array([[0.0, 1.0]] * D)
    Y = Reducers.reduce(data=X, vlim=vlim, nd=D, n_components=k, method=method)
    X_back = Reducers.project(original_data=X, samples=Y, vlim=vlim, nd=D, n_components=k, method=method)
    np.testing.assert_allclose(X_back, X, atol=1e-10)


def test_displacement_lift_returns_base_point_for_zero_displacement():
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(30, 15))
    vlim = np.array([[0.0, 1.0]] * 15)
    red = Reducers(data=X, vlim=vlim, nd=15, n_components=3, method="pod")
    Y = red.rd()
    np.testing.assert_allclose(red.displace_in_original_space(Y[:5], Y[:5], X[:5]), X[:5], atol=1e-12)
    # whereas the mean-anchored lift loses the discarded components
    assert np.linalg.norm(red.project_rd_to_original_space(Y[:5]) - X[:5]) > 0.1


def _interior_reducer(method, D=20, k=3, seed=8):
    # Data well inside the box, so small moves are never clipped
    X = 0.4 + 0.2 * np.random.default_rng(seed).uniform(size=(40, D))
    red = Reducers(data=X, vlim=np.array([[0.0, 1.0]] * D), nd=D, n_components=k, method=method)
    return red, X, red.rd()


@pytest.mark.parametrize("method", ["pca", "pod"])
def test_residual_lift_loses_nothing(method):
    # Property 1: zero reduced-space move and no noise returns the base point
    red, X, Y = _interior_reducer(method)
    good = X[:10]
    lifted = red.residual_lift(Y[:10], Y[:10], good, good, np.random.default_rng(0), noise_scale=0.0)
    np.testing.assert_allclose(lifted, good, atol=1e-12)


@pytest.mark.parametrize("method", ["pca", "pod"])
def test_residual_lift_preserves_reduced_coordinates(method):
    # Property 2: the lifted point projects back exactly onto the candidate
    # y', so l/g computed on y' is the score of the point evaluated - even
    # though the noise moves it in the discarded directions.
    red, X, Y = _interior_reducer(method)
    good = X[:10]
    rng = np.random.default_rng(1)
    y_new = Y[:10] + 0.05 * rng.standard_normal(Y[:10].shape)
    lifted = red.residual_lift(y_new, Y[:10], good, good, rng, noise_scale=0.5)
    np.testing.assert_allclose(red.to_standardized(lifted) @ red.eigenvectors_top_k, y_new, atol=1e-10)
    # ... and the noise does move it off the base point's k-dimensional slice
    off_slice = lifted - red.displace_in_original_space(y_new, Y[:10], good)
    assert np.linalg.norm(off_slice) > 1e-3


def test_residual_lift_explores_all_dimensions():
    # Unlike the anchored lift, every coordinate of the base point moves
    red, X, Y = _interior_reducer("pca", D=30)
    good = X[:10]
    base = np.repeat(good[:1], 200, axis=0)
    yb = np.repeat(Y[:1], 200, axis=0)
    lifted = red.residual_lift(yb, yb, base, good, np.random.default_rng(2))
    assert np.all(lifted.std(axis=0) > 0)
    anchored = red.displace_in_original_space(yb, yb, base)
    assert np.allclose(anchored.std(axis=0), 0)


def test_random_basis_is_orthonormal_and_seeded():
    X = np.random.default_rng(4).uniform(size=(20, 12))
    vlim = np.array([[0.0, 1.0]] * 12)
    bases = []
    for seed in (5, 5, 6):
        red = Reducers(data=X, vlim=vlim, nd=12, n_components=4, method="random", seed=seed)
        red.rd()
        bases.append(red.eigenvectors_top_k)
    np.testing.assert_allclose(bases[0].T @ bases[0], np.eye(4), atol=1e-12)
    np.testing.assert_array_equal(bases[0], bases[1])
    assert not np.allclose(bases[0], bases[2])


@pytest.mark.parametrize("lift", ["mean", "anchored", "residual"])
def test_reduced_suggestions_are_full_dimensional_bounded_and_ranked(lift):
    tpe = make_tpe(lift=lift)
    assert tpe._space_reduced and tpe.n_d == 3
    cands = tpe.resample()
    assert cands.ndim == 2 and cands.shape[1] == 12
    assert np.all((cands >= 0.0) & (cands <= 1.0))
    assert tpe.candidate_ratios.shape == (cands.shape[0],)


@pytest.mark.parametrize("lift", ["mean", "anchored", "residual"])
def test_reduced_suggestions_are_reproducible(lift):
    a = make_tpe(seed=11, lift=lift).resample(seed=3)
    b = make_tpe(seed=11, lift=lift).resample(seed=3)
    np.testing.assert_array_equal(a, b)


def test_anchored_lift_stays_closer_to_good_points():
    def dist_to_good(cands, good):
        return np.mean(np.min(np.linalg.norm(cands[:, None, :] - good[None], axis=2), axis=1))

    gX = make_split(dim=30)[0]
    d_mean = dist_to_good(make_tpe(lift="mean", dim=30, n_r=64).resample(), gX)
    d_anch = dist_to_good(make_tpe(lift="anchored", dim=30, n_r=64).resample(), gX)
    assert d_anch < 0.5 * d_mean


def test_base_points_are_drawn_best_first(monkeypatch):
    tpe = make_tpe(n_r=2000)
    drawn = []

    def record(cls, ks, weights, base_point, vlim, scale=1.0):
        drawn.append(np.asarray(base_point))
        return np.asarray(base_point)

    monkeypatch.setattr(BiTPE, "kde_resample", classmethod(record))
    tpe.resample()
    counts = np.array([sum(np.array_equal(d, row) for d in drawn) for row in tpe.good_obs])
    # Linear rank weights n, n-1, ..., 1: draw counts fall with rank
    expected = np.arange(len(counts), 0, -1)
    assert counts.sum() == 2000
    assert np.corrcoef(counts, expected)[0, 1] > 0.9
    assert counts[0] > 3 * counts[-1]


def test_invalid_lift_is_rejected():
    with pytest.raises(ValueError):
        make_tpe(lift="nearest")
