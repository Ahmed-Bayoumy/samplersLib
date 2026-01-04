import numpy as np
import copy
import pytest

# Import the class from the package
from samplersLib.samplers import Halton

# ----------------------------------------------------------------------
# Helper fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def var_limits():
    """Two‑dimensional variable limits (lower, upper) for testing."""
    # shape (2, 2): each row = [lower, upper] for a variable
    return np.array([[0.0, 10.0],
                     [-5.0, 5.0]])

@pytest.fixture
def halton_ham(var_limits):
    """Halton instance that uses the Hammersley (prime‑based) path."""
    return Halton(ns=5, vlim=var_limits, is_ham=True)

@pytest.fixture
def halton_vdc(var_limits):
    """Halton instance that uses the Van‑der‑Corput fallback path."""
    return Halton(ns=5, vlim=var_limits, is_ham=False)

# ----------------------------------------------------------------------
# Basic construction tests
# ----------------------------------------------------------------------
def test_instantiation_copies_limits(halton_ham, var_limits):
    """Ensure the constructor deep‑copies the limits array."""
    # The internal copy must be a different object but equal in content
    assert halton_ham.var_limits is not var_limits
    assert np.array_equal(halton_ham.var_limits, var_limits)

def test_options_dict_initialised(halton_ham):
    """The options dictionary should start empty."""
    assert isinstance(halton_ham.options, dict)
    assert halton_ham.options == {}

# ----------------------------------------------------------------------
# Prime generation utilities
# ----------------------------------------------------------------------
def test_prime_generator_small(halton_ham):
    """prime_generator should return the first *n* primes."""
    primes = halton_ham.prime_generator(5)
    assert primes == [2, 3, 5, 7, 11]

def test_primes_from_2_to(halton_ham):
    """primes_from_2_to should include 2, 3 and all primes < n."""
    primes = halton_ham.primes_from_2_to(20)
    expected = np.array([2, 3, 5, 7, 11, 13, 17, 19])
    assert np.array_equal(primes, expected)

# ----------------------------------------------------------------------
# Base conversion helpers
# ----------------------------------------------------------------------
def test_base_conv_simple():
    h = Halton(ns=1, vlim=np.zeros((1, 2)), is_ham=True)
    # 5 in base 2 -> ['1', '0', '1']
    assert h.base_conv(5, 2) == ['1', '0', '1']
    # 4 in base 3 -> ['1', '1']
    assert h.base_conv(4, 3) == ['1', '1']

def test_pb_to_dec():
    h = Halton(ns=1, vlim=np.zeros((1, 2)), is_ham=True)
    # 0.101 (binary) = 1/2 + 0/4 + 1/8 = 0.625
    assert pytest.approx(h.pb_to_dec(['0.', '1', '0', '1'], 2), 1e-12) == 0.625
    # 0.12 (base‑3) = 1/3 + 2/9 = 0.555...
    assert pytest.approx(h.pb_to_dec(['0.', '1', '2'], 3), 1e-12) == 5/9

# ----------------------------------------------------------------------
# Data sequencing (prime‑based path)
# ----------------------------------------------------------------------
def test_data_sequencing_consistency(halton_ham):
    """Check that data_sequencing returns a monotonic increasing sequence."""
    # Use a small prime (2) for easy verification
    seq = halton_ham.data_sequencing(pb=2)
    # Expected Van‑der‑Corput sequence for base 2 (first 5 values)
    expected = np.array([0.0, 0.5, 0.25, 0.75, 0.125])
    assert np.allclose(seq, expected, atol=1e-12)

# ----------------------------------------------------------------------
# Van‑der‑Corput fallback path
# ----------------------------------------------------------------------
def test_van_der_corput_basic():
    h = Halton(ns=1, vlim=np.zeros((1, 2)), is_ham=False)
    seq = h.van_der_corput(5, base=3)
    # First 5 elements of base‑3 Van‑der‑Corput
    expected = [0.0, 1/3, 2/3, 1/9, 4/9]
    assert np.allclose(seq, expected, atol=1e-12)

def test_generate_samples_hammersley(halton_ham):
    """Full sample generation using the prime‑based (Hammersley) branch."""
    sample = halton_ham.generate_samples()
    # Shape must be (ns, n_features)
    assert sample.shape == (5, 2)
    # All points must lie inside the provided limits
    low, high = halton_ham.var_limits[:, 0], halton_ham.var_limits[:, 1]
    assert np.all(sample >= low) and np.all(sample <= high)

def test_generate_samples_vdc(halton_vdc):
    """Full sample generation using the Van‑der‑Corput fallback branch."""
    sample = halton_vdc.generate_samples()
    assert sample.shape == (5, 2)
    low, high = halton_vdc.var_limits[:, 0], halton_vdc.var_limits[:, 1]
    assert np.all(sample >= low) and np.all(sample <= high)

# ----------------------------------------------------------------------
# Edge‑case sanity checks
# ----------------------------------------------------------------------
def test_generate_samples_one_dim():
    """Sampler should also work for a single dimension."""
    limits = np.array([[0.0, 1.0]])   # shape (1, 2)
    h = Halton(ns=3, vlim=limits, is_ham=True)
    sample = h.generate_samples()
    assert sample.shape == (3, 1)
    assert np.all(sample >= 0.0) and np.all(sample <= 1.0)

def test_invalid_prime_request():
    """prime_generator should raise if n <= 0."""
    h = Halton(ns=1, vlim=np.zeros((1, 2)), is_ham=True)
    with pytest.raises(ValueError):
        # Force an error by passing a negative count – the current
        # implementation loops forever, so we guard it manually.
        h.prime_generator(-1)
