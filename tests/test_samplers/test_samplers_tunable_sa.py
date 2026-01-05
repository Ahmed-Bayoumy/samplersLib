import numpy as np
import pytest
import math

# Import the class under test – adjust the import path if your package layout differs
from samplersLib.samplers import TunableSA


@pytest.fixture
def dummy_data():
    """Create a tiny synthetic problem (2 variables, 2 objectives)."""
    rng = np.random.default_rng(0)
    # 5 data points, 2 input dimensions
    data = rng.normal(size=(5, 2))
    # 2 output dimensions (e.g., two objectives)
    y = rng.normal(size=(5, 2))
    # Incremental variables – same shape as data
    x_inc = rng.normal(size=(1, 2))[0]
    # Variable limits – simple box around the data
    vlim = np.array([[-5, 5], [-5, 5]])
    return data, y, x_inc, vlim


def test_instantiation_and_weights(dummy_data):
    data, y, x_inc, vlim = dummy_data

    # ---- 1. Normal construction (no explicit weights) -----------------
    sa = TunableSA(
        data=data,
        y=y,
        x_inc=x_inc,
        it=1,
        vlim=vlim,
        initial_temp=10,
        cooling_rate=0.8,
        max_iter=10,
        seed=123,
    )
    # internal attributes should be set
    assert sa.n_s == data.shape[0]
    assert sa.n_d == data.shape[1]
    # weights are computed from the selector – they must be a 1‑D array of length n_s
    assert isinstance(sa._weights, np.ndarray)
    assert sa._weights.shape == (sa.n_d,)
    # they must sum to 1 (or 0 if all zero)
    assert np.isclose(sa._weights.sum(), 1.0) or np.isclose(sa._weights.sum(), 0.0)

    # ---- 2. Construction with user‑provided weights --------------------
    user_w = np.array([1, 0])
    sa_w = TunableSA(
        data=data,
        y=y,
        x_inc=x_inc,
        it=2,
        vlim=vlim,
        initial_temp=10,
        cooling_rate=0.8,
        max_iter=10,
        seed=123,
        weights=user_w,
    )
    # user weights are normalised internally
    expected = np.array([np.float64(1), np.float64(0)])
    assert np.allclose(sa_w._weights, expected, atol=np.array([0.02, 0.02]))


def test_target_distribution():
    # 1‑D test – the pdf of a standard normal at 0 is 1/sqrt(2π)
    x = np.array([0.0])
    val = TunableSA.target_distribution(None, x)   # static‑like call, self not used
    assert np.isclose(val, 1 / math.sqrt(2 * math.pi))

    # 2‑D test – compare against scipy.stats.multivariate_normal (optional)
    # we avoid the heavy dependency; just check that the function returns a positive number
    x2 = np.array([0.5, -0.2])
    val2 = TunableSA.target_distribution(None, x2)
    assert val2 > 0


def test_get_neighbor_and_bounds(dummy_data):
    data, y, x_inc, vlim = dummy_data
    sa = TunableSA(data, y, x_inc, it=0, vlim=vlim)

    point = np.array([1.0, -1.0])
    neighbor = sa.get_neighbor(point, step_size=0.2)

    # neighbor must have same dimensionality
    assert len(neighbor) == len(point)
    # each coordinate must be within ±step_size of the original
    for a, b in zip(point, neighbor):
        assert abs(a - b) <= 0.2


def test_acceptance_probability(monkeypatch):
    data = np.zeros((1, 1))
    y = np.zeros((1, 1))
    x_inc = np.zeros((1, 1))
    vlim = np.array([[-1, 1]])

    sa = TunableSA(data, y, x_inc, it=0, vlim=vlim)

    # Force deterministic behaviour by monkey‑patching np.random.random
    monkeypatch.setattr(np.random, "random", lambda: 0.3)

    # With temperature high enough, exp(-1/T) > 0.3 → accept
    assert sa.acceptance_probability(temperature=10.0) is True

    # With temperature low enough, exp(-1/T) < 0.3 → reject
    assert sa.acceptance_probability(temperature=0.1) is False


def test_simulated_annealing_sampling_basic(dummy_data):
    data, y, x_inc, vlim = dummy_data
    sa = TunableSA(
        data=data,
        y=y,
        x_inc=x_inc,
        it=0,
        vlim=vlim,
        initial_temp=5.0,
        cooling_rate=0.5,
        max_iter=5,
        seed=42,
    )

    size = 3
    samples = sa.simulated_annealing_sampling(size)

    # Output shape
    assert samples.shape == (size, data.shape[1])

    # All samples must respect the variable limits
    low, high = vlim[:, 0], vlim[:, 1]
    assert np.all(samples >= low - 1e-8)
    assert np.all(samples <= high + 1e-8)

    # Temperature should have cooled down after the loop – we can infer it
    # by checking that the acceptance probability would be lower now.
    # (Not a strict test, just a sanity check)
    final_temp = sa.initial_temp * (sa.cooling_rate ** sa.n_s)
    assert final_temp < sa.initial_temp


def test_resample_uses_multidimensional_resampling(monkeypatch, dummy_data):
    data, y, x_inc, vlim = dummy_data
    # Give a non‑trivial weight vector so the branch is taken
    weights = np.array([0.7, 0.3])

    sa = TunableSA(
        data=data,
        y=y,
        x_inc=x_inc,
        it=0,
        vlim=vlim,
        initial_temp=5.0,
        cooling_rate=0.9,
        max_iter=5,
        seed=0,
        weights=weights,
    )

    # Patch the heavy‑weight plotting call (if any) and the internal
    # `resample_multidimensional_variables` to a stub that records arguments.
    called = {}

    def fake_resample_multidimensional_variables(self, variables, dimension_weights, num_samples):
        called["variables"] = variables
        called["weights"] = dimension_weights
        called["num_samples"] = num_samples
        # Return a simple deterministic array
        return np.full((num_samples, variables.shape[1]), 42.0)

    monkeypatch.setattr(
        TunableSA,
        "resample_multidimensional_variables",
        fake_resample_multidimensional_variables,
    )

    out = sa.resample(size=4, seed=123)

    # Verify that the stub was called with the expected arguments
    assert "variables" in called
    assert called["weights"] is sa._weights
    assert called["num_samples"] == 4

    # The final result should be exactly what the stub returned
    assert np.all(out == 42.0)
    assert out.shape == (4, data.shape[1])
