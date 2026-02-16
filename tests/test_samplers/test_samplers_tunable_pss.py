import numpy as np
import pytest

# Import the class under test
from samplersLib.samplers import TunablePSS


# Helper to create a tiny synthetic problem
def make_dummy_data(n_samples=5, n_dims=3, n_targets=2):
    rng = np.random.default_rng(0)
    data = rng.uniform(0, 1, size=(n_samples, n_dims))
    y = rng.normal(size=(n_samples, n_targets))
    # variable limits: [low, high] for each dimension
    vlim = np.column_stack((np.zeros(n_dims), np.ones(n_dims)))
    return data, y, vlim


@pytest.fixture
def sampler():
    data, y, vlim = make_dummy_data()
    # small particle swarm for speed
    return TunablePSS(
        data=data,
        y=y,
        x_inc=data[-1],  # use the same points as incumbents
        it=0,
        vlim=vlim,
        num_particles=10,
        max_iter=5,
        inertia_weight=0.5,
        cognitive_weight=1.0,
        social_weight=1.0,
        seed=42,
    )


def test_initialisation_weights(sampler):
    # weights should be created from the RF selectors and sum to 1
    assert hasattr(sampler, "_weights")
    assert np.isclose(sampler._weights.sum(), 1.0)


def test_target_distribution(sampler):
    x = np.zeros(sampler.data.shape[1])
    val = sampler.target_distribution(x)
    # Gaussian density at the mean must be positive and equal to the normalising constant
    expected = 1.0 / np.sqrt((2 * np.pi) ** len(x))
    assert np.isclose(val, expected)


def test_particle_swarm_sampling_limits(sampler):
    size = 8
    samples = sampler.particle_swarm_sampling(size)

    # shape check
    assert samples.shape == (size, sampler.n_d)

    # every sample must lie inside the variable limits
    low, high = sampler.var_limits
    assert np.all(samples >= low - 1e-12)
    assert np.all(samples <= high + 1e-12)


def test_resample_calls_pso_and_weights(sampler):
    size = 12

    # monkey‑patch the heavy PSO method to make the test deterministic & fast
    def fake_pso(_size):
        # return a deterministic grid inside the limits
        grid = np.linspace(0, 1, _size).reshape(-1, 1)
        return np.repeat(grid, sampler.n_d, axis=1)

    sampler.particle_swarm_sampling = fake_pso

    res = sampler.resample(size=size, seed=123)

    # result shape must match the requested size (or fewer after uniquing)
    assert res.shape[1] == sampler.n_d
    assert res.shape[0] <= size

    # when weights are non‑zero the method should have invoked the
    # resample_multidimensional_variables logic – we can verify that the
    # returned values are drawn from the fake PSO grid
    assert np.all(np.isin(res[:, 0], np.linspace(0, 1, size)))


def test_resample_without_weights():
    # Build a sampler with explicit zero weights to trigger the fallback path
    data, y, vlim = make_dummy_data()
    sampler = TunablePSS(
        data=data,
        y=y,
        x_inc=data,
        it=0,
        vlim=vlim,
        num_particles=5,
        max_iter=3,
        weights=np.zeros(data.shape[1]),  # force zero‑weight branch
    )
    # patch PSO to a simple deterministic output
    sampler.particle_swarm_sampling = lambda sz: np.full((sz, data.shape[1]), 0.5)

    out = sampler.resample(size=4, seed=0)
    # With zero weights the method should return the raw PSO samples unchanged
    assert np.allclose(out, 0.5)
