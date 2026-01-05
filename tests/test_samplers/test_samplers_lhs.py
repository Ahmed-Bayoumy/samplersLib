import numpy as np
import copy
import pytest

# Import the class from your package – adjust the import path if needed
from samplersLib.samplers import LHS


@pytest.fixture
def var_limits():
    """Two‑dimensional unit‑cube limits."""
    return np.array([[0.0, 1.0], [0.0, 1.0]])


def test_instantiation_defaults(var_limits):
    """Basic construction should store the arguments correctly."""
    lhs = LHS(ns=10, vlim=copy.deepcopy(var_limits))
    assert lhs.n_s == 10
    # internal copy – mutating the original must not affect the instance
    var_limits[0, 0] = -1.0
    assert lhs.var_limits[0, 0] == 0.0
    # default options
    assert lhs.options["criterion"] == "ExactSE"
    assert lhs.options["randomness"] == 10000


def test_set_options():
    """Changing options must be reflected in the internal dict."""
    lhs = LHS(ns=5, vlim=np.array([[0, 1]]))
    lhs.set_options(c="maximin", r=42)
    assert lhs.options["criterion"] == "maximin"
    assert lhs.options["randomness"] == 42


def test_generate_samples_basic(var_limits):
    """`generate_samples` should return an array of the requested shape
    and respect the variable limits."""
    np.random.seed(0)          # make the test deterministic
    lhs = LHS(ns=7, vlim=var_limits)
    lhs.set_options(c="center", r=123)   # any non‑ExactSE criterion
    samples = lhs.generate_samples()
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (7, var_limits.shape[0])

    # each column must lie inside its bounds
    for i, (low, high) in enumerate(var_limits):
        assert np.all(samples[:, i] >= low)
        assert np.all(samples[:, i] <= high)


def test_generate_samples_exactse(var_limits):
    """When the criterion is 'ExactSE' the code follows the exact‑SE branch."""
    np.random.seed(1)
    lhs = LHS(ns=6, vlim=var_limits)
    # keep the default criterion = ExactSE
    samples = lhs.generate_samples()
    assert samples.shape == (6, var_limits.shape[0])
    # values must still be inside the limits
    assert np.all(samples >= var_limits[:, 0])
    assert np.all(samples <= var_limits[:, 1])


def test_expand_lhs_basic(var_limits):
    """`expand_lhs` should add the requested number of points."""
    np.random.seed(2)
    lhs = LHS(ns=4, vlim=var_limits)
    base = lhs.generate_samples()
    expanded = lhs.expand_lhs(base, n_points=3, method="basic")
    # original points + new points
    assert expanded.shape[0] == base.shape[0] + 3
    # still within limits
    assert np.all(expanded >= var_limits[:, 0])
    assert np.all(expanded <= var_limits[:, 1])


def test_expand_lhs_exactse(var_limits):
    """`expand_lhs` with method='ExactSE' must still return a correctly‑shaped array."""
    np.random.seed(3)
    lhs = LHS(ns=5, vlim=var_limits)
    base = lhs.generate_samples()
    expanded = lhs.expand_lhs(base, n_points=2, method="ExactSE")
    assert expanded.shape[0] == base.shape[0] + 2
    # sanity check – the returned array is still inside the limits
    assert np.all(expanded >= var_limits[:, 0])
    assert np.all(expanded <= var_limits[:, 1])
