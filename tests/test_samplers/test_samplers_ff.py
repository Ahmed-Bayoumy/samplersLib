import copy
import numpy as np
import pytest

# Import the class you want to test
from samplersLib.samplers import FullFactorial


@pytest.fixture
def simple_limits():
    """Two‑dimensional limits: x ∈ [0, 10], y ∈ [‑5, 5]"""
    return np.array([[0.0, 10.0],
                     [-5.0, 5.0]])


def test_instantiation_default_weights(simple_limits):
    """When ``weights`` is None the sampler should treat all dimensions equally."""
    sampler = FullFactorial(ns=4, w=None, c=False, vlim=simple_limits)

    # internal state should reflect a deep copy of the limits
    assert np.array_equal(sampler.var_limits, simple_limits)
    # weights attribute should be None (handled later in generate_samples)
    assert sampler.options["weights"] is None


def test_set_options_updates_internal_state(simple_limits):
    """set_options must replace the internal options dict and copy limits."""
    sampler = FullFactorial(ns=4, w=None, c=False, vlim=simple_limits)

    new_weights = np.array([0.7, 0.3])
    new_limits = np.array([[1.0, 2.0], [3.0, 4.0]])
    sampler.set_options(w=new_weights, c=True, la=new_limits)

    # options dict should contain the new values (deep‑copied)
    assert np.array_equal(sampler.options["weights"], new_weights)
    assert sampler.options["clip"] is True
    assert np.array_equal(sampler.options["limits"], new_limits)

    # original limits must stay unchanged
    assert np.array_equal(sampler.var_limits, simple_limits)


def test_generate_samples_no_clip_equal_weights(simple_limits):
    """With equal weights and ``clip=True`` the grid size may be larger than ns."""
    sampler = FullFactorial(ns=5, w=None, c=True, vlim=simple_limits)

    samples = sampler.generate_samples()

    # Because the algorithm adds levels until prod(num_list) >= ns,
    # the smallest grid that satisfies ns=5 is 2×3 = 6 points.
    assert samples.shape == (6, 2)

    # Verify scaling: first column should be in [0,10], second in [-5,5]
    assert samples[:, 0].min() >= 0.0 and samples[:, 0].max() <= 10.0
    assert samples[:, 1].min() >= -5.0 and samples[:, 1].max() <= 5.0


def test_generate_samples_with_clip_and_weights(simple_limits):
    """When ``clip=True`` the returned array must contain exactly ns points."""
    # Give a strong weight to the first dimension → more levels there
    weights = np.array([0.9, 0.1])
    sampler = FullFactorial(ns=7, w=weights, c=True, vlim=simple_limits)

    samples = sampler.generate_samples()

    # Clip forces the output size to exactly ns
    assert samples.shape == (7, 2)

    # Check that the points are still within the limits
    assert np.all(samples[:, 0] >= simple_limits[0, 0])
    assert np.all(samples[:, 0] <= simple_limits[0, 1])
    assert np.all(samples[:, 1] >= simple_limits[1, 0])
    assert np.all(samples[:, 1] <= simple_limits[1, 1])


def test_generate_samples_correct_scaling(simple_limits):
    """The unit‑cube points must be linearly mapped to the provided limits."""
    sampler = FullFactorial(ns=4, w=None, c=False, vlim=simple_limits)

    # Force a known grid: 2 levels per dimension → points at 0 and 1 in unit space
    # (the algorithm already does this for ns=4)
    samples = sampler.generate_samples()

    # Expected scaled values
    expected = np.array([
        [0.0, -5.0],
        [0.0,  5.0],
        [10.0, -5.0],
        [10.0,  5.0],
    ])

    # Order may differ because of meshgrid ordering; sort rows for comparison
    assert np.allclose(np.sort(samples, axis=0), np.sort(expected, axis=0))
    