import numpy as np
import copy
import pytest

# Import the class under test
from samplersLib.samplers import RS

@pytest.fixture
def var_limits():
    """Create a simple 2‑dimensional limit matrix."""
    # each row = [lower, upper] for a variable
    return np.array([[0.0, 1.0],
                     [-5.0, 5.0]])

def test_instantiation(var_limits):
    """RS should store the arguments unchanged."""
    opts = {"randomness": 42}
    sampler = RS(ns=10, vlim=var_limits, options=opts)

    # n_s and var_limits are stored correctly
    assert sampler.n_s == 10
    assert np.array_equal(sampler.var_limits, var_limits)

    # options are deep‑copied (modifying the original dict shouldn't affect the instance)
    opts["new_key"] = "value"
    assert "new_key" not in sampler.options

def test_generate_samples_shape_and_limits(var_limits):
    """Generated samples must have shape (ns, n_variables) and lie inside limits."""
    sampler = RS(ns=7, vlim=var_limits)

    samples = sampler.generate_samples()
    assert samples.shape == (7, var_limits.shape[0])

    # each column must be within its corresponding limits
    lower = var_limits[:, 0]
    upper = var_limits[:, 1]
    assert np.all(samples >= lower) and np.all(samples <= upper)

def test_generate_samples_reproducibility(var_limits):
    """Providing a seed via options should make the output deterministic."""
    seed = 12345
    opts = {"randomness": seed}
    sampler1 = RS(ns=5, vlim=var_limits, options=opts)
    sampler2 = RS(ns=5, vlim=var_limits, options=opts)

    samples1 = sampler1.generate_samples()
    samples2 = sampler2.generate_samples()

    # With the same seed the two calls must produce identical arrays
    np.testing.assert_array_equal(samples1, samples2)

def test_placeholder_methods_do_not_raise(var_limits):
    """methods, utilities and set_options currently do nothing – they should be callable."""
    sampler = RS(ns=3, vlim=var_limits)

    # No exception should be raised
    sampler.methods()
    sampler.utilities()
    sampler.set_options()
