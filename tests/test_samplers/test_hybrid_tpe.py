"""
Tests for the adaptive hybrid of reduced-space TPE and a model-free trust
region (`samplersLib.hybrid`).
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from samplersLib.hybrid import AdaptiveHybridTPE, CholeskyKDE, ThompsonAllocator


# pylint: disable=missing-function-docstring
def shifted_sphere(x, shift):
    return float(np.sum((np.asarray(x) - shift) ** 2))


def test_cholesky_kde_matches_gaussian_mixture():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(12, 3)) @ np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 0.4]])
    kde = CholeskyKDE(data)
    X = rng.normal(size=(5, 3))
    H = kde.L @ kde.L.T
    expected = np.log(np.mean([multivariate_normal(mean=d, cov=H).pdf(X) for d in data], axis=0))
    np.testing.assert_allclose(kde.log_pdf(X), expected, rtol=1e-10)


def test_cholesky_kde_is_finite_with_fewer_points_than_dimensions():
    # 10 points in 40 dimensions: the sample covariance is singular, the
    # shrunk one is not, so densities near the data stay finite
    data = np.random.default_rng(1).uniform(size=(10, 40))
    kde = CholeskyKDE(data)
    assert np.all(np.isfinite(kde.log_pdf(data + 1e-3)))


def test_thompson_allocator_prefers_the_successful_arm_and_forgets():
    alloc = ThompsonAllocator(np.random.default_rng(2), p_min=0.1, discount=0.98)
    for _ in range(50):
        alloc.update(0, 0.0)
        alloc.update(1, 1.0)
    picks = [alloc.choose() for _ in range(2000)]
    assert 0.85 < np.mean(picks) < 1.0  # p_min keeps arm 0 alive
    for _ in range(300):
        alloc.update(0, 1.0)
        alloc.update(1, 0.0)
    assert np.mean([alloc.choose() for _ in range(2000)]) < 0.15  # old evidence was discounted away


def test_points_stay_in_bounds_and_initial_points_come_first():
    vlim = np.array([[-5.0, 5.0], [0.0, 100.0], [10.0, 11.0], [-1.0, 0.0]])
    init = np.array([[0.0, 50.0, 10.5, -0.5], [1.0, 20.0, 10.1, -0.2], [-3.0, 80.0, 10.9, -0.9]])
    opt = AdaptiveHybridTPE(vlim, initial_points=init, seed=3)
    for i in range(3):
        x = opt.ask()
        np.testing.assert_allclose(x, init[i])
        opt.tell(x, shifted_sphere(x, 0.0))
    for _ in range(40):
        x = opt.ask()
        assert np.all(x >= vlim[:, 0]) and np.all(x <= vlim[:, 1])
        opt.tell(x, shifted_sphere(x, 0.0))
    assert len(opt.F) == 43 and len(opt.arms) == 40


def test_runs_are_reproducible_and_scale_invariant():
    shift = np.linspace(-2, 2, 6)
    vlim = np.array([[-5.0, 5.0]] * 6)
    a = AdaptiveHybridTPE(vlim, n_init=10, seed=4)
    a.minimize(lambda x: shifted_sphere(x, shift), 60)
    b = AdaptiveHybridTPE(vlim, n_init=10, seed=4)
    b.minimize(lambda x: shifted_sphere(x, shift), 60)
    np.testing.assert_array_equal(a.F, b.F)
    # the same problem posed on the unit cube gives the same run
    c = AdaptiveHybridTPE(np.array([[0.0, 1.0]] * 6), n_init=10, seed=4)
    c.minimize(lambda u: shifted_sphere(-5.0 + 10.0 * np.asarray(u), shift), 60)
    np.testing.assert_allclose(c.F, a.F, rtol=1e-9, atol=1e-12)


def test_minimize_beats_random_search():
    D, budget = 10, 80
    vlim = np.array([[-5.0, 5.0]] * D)
    shift = np.random.default_rng(5).uniform(-3, 3, D)
    hybrid, random = [], []
    for seed in range(3):
        opt = AdaptiveHybridTPE(vlim, n_init=20, seed=seed)
        _, f_best = opt.minimize(lambda x: shifted_sphere(x, shift), budget)
        assert f_best < opt.F[:20].min()
        hybrid.append(f_best)
        X = np.random.default_rng(seed).uniform(-5, 5, (budget, D))
        random.append(min(shifted_sphere(x, shift) for x in X))
    assert np.median(hybrid) < 0.5 * np.median(random)


def test_ask_tell_protocol_is_enforced():
    opt = AdaptiveHybridTPE(np.array([[0.0, 1.0]] * 3), n_init=3, seed=6)
    with pytest.raises(RuntimeError):
        opt.tell(np.zeros(3), 1.0)
    x = opt.ask()
    np.testing.assert_array_equal(opt.ask(), x)  # repeated ask returns the pending point
    with pytest.raises(ValueError):
        opt.tell(x + 0.1, 1.0)
    with pytest.raises(ValueError):
        opt.tell(x, np.nan)
    opt.tell(x, 1.0)


@pytest.mark.parametrize("n_candidates", [1, 8, 256])
def test_candidate_pool_size_sets_how_many_candidates_are_scored(n_candidates, monkeypatch):
    import samplersLib.hybrid as hybrid

    sizes = []
    original = hybrid.CholeskyKDE.log_pdf

    def counting_log_pdf(self, X):
        sizes.append(len(np.atleast_2d(X)))
        return original(self, X)

    monkeypatch.setattr(hybrid.CholeskyKDE, "log_pdf", counting_log_pdf)
    opt = AdaptiveHybridTPE(np.array([[0.0, 1.0]] * 5), n_init=10, seed=7, n_candidates=n_candidates)
    opt.minimize(lambda x: shifted_sphere(x, 0.3), 40)
    assert len(opt.F) == 40 and np.all((opt.X >= 0) & (opt.X <= 1))
    # every local-arm proposal scored exactly the pool (for l and for g)
    assert sizes and set(sizes) == {n_candidates}


def test_uniform_safeguard_is_off_by_default_and_uniform_when_on():
    vlim = np.array([[-2.0, 3.0]] * 4)

    def f(x):
        return shifted_sphere(x, 0.5)

    base = AdaptiveHybridTPE(vlim, n_init=6, seed=9)
    base.minimize(f, 30)
    off = AdaptiveHybridTPE(vlim, n_init=6, seed=9, uniform_prob=0.0)
    off.minimize(f, 30)
    np.testing.assert_array_equal(base.F, off.F)  # rho = 0 changes nothing
    assert off.safeguard_steps == 0

    # rho = 1: every post-design point is the safeguard's uniform draw
    on = AdaptiveHybridTPE(vlim, n_init=6, seed=9, uniform_prob=1.0)
    on.minimize(f, 30)
    rng = np.random.default_rng(9 + 37)
    expected = []
    for _ in range(24):
        rng.random()
        expected.append(rng.random(4))
    np.testing.assert_allclose(on.X[6:], -2.0 + 5.0 * np.array(expected))
    assert on.safeguard_steps == 24

    with pytest.raises(ValueError):
        AdaptiveHybridTPE(vlim, uniform_prob=1.5)


def test_halton_safeguard_places_halton_points_on_a_fixed_schedule():
    from scipy.stats import qmc

    vlim = np.array([[-2.0, 3.0]] * 4)

    def f(x):
        return shifted_sphere(x, 0.5)

    base = AdaptiveHybridTPE(vlim, n_init=6, seed=11)
    base.minimize(f, 40)
    off = AdaptiveHybridTPE(vlim, n_init=6, seed=11, halton_every=0)
    off.minimize(f, 40)
    np.testing.assert_array_equal(base.F, off.F)  # disabled by default, nothing changes

    m = 5
    on = AdaptiveHybridTPE(vlim, n_init=6, seed=11, halton_every=m)
    on.minimize(f, 40)
    halton = qmc.Halton(d=4, scramble=False)
    halton.fast_forward(1)
    expected = -2.0 + 5.0 * halton.random(34 // m)
    # post-design evaluations m, 2m, ... (1-based) are the Halton points, in order
    np.testing.assert_allclose(on.X[6:][m - 1::m], expected)
    assert on.safeguard_steps == 34 // m
    # the same points whatever the seed: the safeguard does not depend on the run
    other = AdaptiveHybridTPE(vlim, n_init=6, seed=12, halton_every=m)
    other.minimize(f, 40)
    np.testing.assert_allclose(other.X[6:][m - 1::m], expected)

    with pytest.raises(ValueError):
        AdaptiveHybridTPE(vlim, halton_every=-1)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(vlim=np.array([[1.0, 0.0]])),
        dict(vlim=np.array([0.0, 1.0])),
        dict(vlim=np.array([[0.0, 1.0]] * 2), n_init=2),
        dict(vlim=np.array([[0.0, 1.0]] * 2), n_candidates=0),
        dict(vlim=np.array([[0.0, 1.0]] * 2), n_candidates=2.5),
        dict(vlim=np.array([[0.0, 1.0]] * 2), decrease_scale="median"),
        dict(vlim=np.array([[0.0, 1.0]] * 2), recenter_delta="halve"),
        dict(vlim=np.array([[0.0, 1.0]] * 2), local_metric="full"),
        dict(vlim=np.array([[0.0, 1.0]] * 2), metric_rank=0),
        dict(vlim=np.array([[0.0, 1.0]] * 2), restart="random"),
        dict(vlim=np.array([[0.0, 1.0]] * 2), step_control="cma"),
        dict(vlim=np.array([[0.0, 1.0]] * 2), local_center="good"),
        dict(vlim=np.array([[0.0, 1.0]] * 2), success_damping=0.0),
        dict(vlim=np.array([[0.0, 1.0]] * 2), n_reduced_dims=0),
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        AdaptiveHybridTPE(**kwargs)


LOCAL_OPTIONS = dict(
    decrease_scale="local",
    recenter_delta="keep",
    local_metric="lowrank",
    restart="diverse",
    step_control="success",
    local_center="current",
)


@pytest.mark.parametrize("option", list(LOCAL_OPTIONS))
def test_local_options_stay_in_bounds_and_beat_random_search(option):
    dim, budget = 6, 120
    shift = np.full(dim, 0.3)
    vlim = np.array([[-1.0, 1.0]] * dim)
    hybrid, random = [], []
    for seed in range(3):
        opt = AdaptiveHybridTPE(vlim, seed=seed, **{option: LOCAL_OPTIONS[option]})
        _, f_best = opt.minimize(lambda x: shifted_sphere(x, shift), budget)
        assert np.all(opt.X >= -1.0) and np.all(opt.X <= 1.0) and opt.failures == 0
        hybrid.append(f_best)
        X = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(budget, dim))
        random.append(min(shifted_sphere(x, shift) for x in X))
    assert np.median(hybrid) < np.median(random)


def test_lowrank_metric_learns_the_direction_of_accepted_steps():
    opt = AdaptiveHybridTPE(np.array([[0.0, 1.0]] * 4), local_metric="lowrank", metric_rank=1)
    direction = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2)
    for scale in np.linspace(0.5, 1.0, 12):
        opt._update_metric(scale * direction)  # pylint: disable=protected-access
    A = opt._metric_factor()  # pylint: disable=protected-access
    C = A @ A.T
    w, V = np.linalg.eigh(C)
    assert abs(V[:, -1] @ direction) > 0.99  # leading axis follows the steps
    assert w[-1] > 10 * w[0]


def test_keep_recenter_does_not_reset_the_radius_on_a_global_improvement():
    vlim = np.array([[0.0, 1.0]] * 3)
    for mode, expect_reset in (("reset", True), ("keep", False)):
        opt = AdaptiveHybridTPE(vlim, n_init=5, seed=0, recenter_delta=mode)
        for _ in range(5):
            x = opt.ask()
            opt.tell(x, shifted_sphere(x, 0.5))
        opt.delta = 0.05
        opt._pending = (np.full(3, 0.5), 1)  # pylint: disable=protected-access
        opt.tell(np.full(3, 0.5), 0.0)  # a global proposal at the optimum
        assert (opt.delta == opt.delta_init) == expect_reset


def test_success_rule_grows_the_radius_on_success_and_shrinks_it_on_failure():
    vlim = np.array([[0.0, 1.0]] * 4)
    opt = AdaptiveHybridTPE(vlim, n_init=5, seed=0, step_control="success")
    for _ in range(5):
        x = opt.ask()
        opt.tell(x, shifted_sphere(x, 0.5))
    opt.delta = 0.1
    for f in (-1.0, -2.0, -3.0):  # three successful local steps
        u = np.clip(opt._xk + 0.01, 0, 1)  # pylint: disable=protected-access
        opt._update_trust_region(u, f)  # pylint: disable=protected-access
    grown = opt.delta
    assert grown > 0.1
    for _ in range(40):  # failures: the smoothed success rate falls below the target, then delta shrinks
        opt._update_trust_region(np.clip(opt._xk - 0.01, 0, 1), 10.0)  # pylint: disable=protected-access
    assert opt.delta < grown


def test_step_metric_and_radius_stay_well_defined_along_a_run():
    # Lemma "well-posedness of the step metric and of the radius": C is
    # positive definite, delta stays in [delta_min, delta_max] at the end of
    # every iteration and the success rate stays in [0, 1]
    dim = 8
    vlim = np.array([[-5.0, 5.0]] * dim)
    shift = np.random.default_rng(7).uniform(-3, 3, dim)
    opt = AdaptiveHybridTPE(vlim, n_init=10, seed=7)
    for _ in range(200):
        x = opt.ask()
        opt.tell(x, shifted_sphere(x, shift) + 10.0 * np.sum(np.cos(3 * x)))
        assert opt.delta_min <= opt.delta <= opt.delta_max
        assert 0.0 <= opt._p_succ <= 1.0  # pylint: disable=protected-access
        A = opt._metric_factor()  # pylint: disable=protected-access
        assert np.linalg.eigvalsh(A @ A.T).min() > 0
