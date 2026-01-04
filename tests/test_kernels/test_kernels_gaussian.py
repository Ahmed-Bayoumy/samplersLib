# test_kernels.py
import copy
import numpy as np
import pytest

# Import the class under test
from samplersLib.kernels import Gaussian, KERNEL_TYPE


@pytest.fixture
def sample_data():
    """Create a small deterministic dataset."""
    rng = np.random.default_rng(0)
    return rng.normal(loc=0.0, scale=1.0, size=(5, 2))


def test_instantiation_defaults(sample_data):
    """Basic construction – defaults should set sensible attributes."""
    g = Gaussian(data=sample_data, calculate_bw=False)

    # Data should be deep‑copied
    assert not np.shares_memory(g.data, sample_data)

    # Dimensionality
    assert g._ns == sample_data.shape[0]
    assert g._nd == sample_data.shape[1]

    # Effective sample size defaults to number of samples
    assert g._ne == g._ns

    # Bandwidth method stored correctly
    assert g.bw_method == "SCOTT"
    assert g._type == KERNEL_TYPE.NONPARAMETRIC

    # No covariance supplied yet
    assert g._cov is None


def test_instantiation_with_ne_and_weights(sample_data):
    """When a positive n_r is supplied it overrides the weight‑based ESS."""
    g = Gaussian(data=sample_data, n_r=3, weights=np.ones(5) / 5)

    # n_r takes precedence
    assert g._ne == 3

    # Bandwidth still not set – calling a kernel method should raise
    with pytest.raises(ValueError):
        g.kf_univar(np.array([0.0]))


def test_bandwidth_setting_and_univariate_kernel(sample_data):
    """Check that a scalar bandwidth works and that the kernel evaluates correctly."""
    bw = 0.5
    g = Gaussian(data=sample_data, h=bw)

    # Univariate kernel on a scalar array
    u = np.array([0.0, bw, -bw, 2 * bw])
    vals = g.kf_univar(u)

    # Expected analytic values
    expected = (1.0 / (np.sqrt(2 * np.pi) * bw)) * np.exp(-0.5 * (u / bw) ** 2)

    np.testing.assert_allclose(vals, expected, rtol=1e-12)

    # Symmetry check
    assert vals[1] == pytest.approx(vals[2])


def test_univariate_kernel_missing_bandwidth(sample_data):
    """Calling kf_univar without a bandwidth should raise."""
    g = Gaussian(data=sample_data, h=None)
    with pytest.raises(ValueError, match="Bandwidth `h` must be set."):
        g.kf_univar(np.array([0.0]))


def test_multivariate_kernel_full_covariance(sample_data):
    """When a full covariance matrix is supplied the multivariate path is used."""
    # Create a simple diagonal covariance for which the analytic result is known
    cov = np.diag([0.25, 0.36])          # variances = 0.25, 0.36  → std = 0.5, 0.6
    g = Gaussian(data=sample_data, h=[0.5, 0.6])
    g._cov = cov                         # set directly for the test

    # Test point at the origin (z = 0) → exponent = 0
    z = np.zeros(2)
    val = g.kf_multivar(z)

    det = np.linalg.det(cov)
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** 2 * det)
    expected = norm_const  # exp(0) = 1

    assert val == pytest.approx(expected, rel=1e-12)

    # Test a non‑zero point and compare against the explicit formula
    z = np.array([0.5, -0.6])
    inv_cov = np.linalg.inv(cov)
    exponent = -0.5 * z.T @ inv_cov @ z
    expected = norm_const * np.exp(exponent)

    np.testing.assert_allclose(g.kf_multivar(z), expected, rtol=1e-12)


def test_multivariate_kernel_diagonal_fallback(sample_data):
    """When no covariance is set the fallback uses the per‑dimension bandwidths."""
    bw = [0.5, 0.6]
    g = Gaussian(data=sample_data, h=bw, calculate_bw=False)

    # No covariance supplied → fallback path
    z = np.array([0.5, -0.6])
    h_arr = np.asarray(bw)
    scaled_z = z / h_arr
    norm_const = np.prod(1 / (np.sqrt(2 * np.pi) * h_arr))
    expected = norm_const * np.exp(-0.5 * np.sum(scaled_z ** 2))

    np.testing.assert_allclose(g.kf_multivar(z), expected, rtol=1e-12)


def test_multivariate_kernel_missing_bandwidth(sample_data):
    """If both covariance and bandwidth are missing, an error is raised."""
    g = Gaussian(data=sample_data, h=None)
    g._cov = None
    with pytest.raises(ValueError, match="Bandwidth `h` must be set for fallback"):
        g.kf_multivar(np.array([0.0, 0.0]))

