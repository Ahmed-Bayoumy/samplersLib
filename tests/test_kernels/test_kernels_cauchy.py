from math import gamma

import numpy as np
import pytest

from samplersLib.kernels import KERNEL_TYPE, Cauchy


@pytest.fixture
def sample_data():
    """Create a small deterministic dataset (5 samples, 2 dimensions)."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(5, 2))


def test_instantiation_basic(sample_data):
    """Kernel should be instantiated without errors and set internal flags."""
    k = Cauchy(data=sample_data, vlim=[[0, 1], [0, 1]], calculate_bw=False)

    # basic attributes
    assert isinstance(k, Cauchy)
    assert k._ns == sample_data.shape[0]
    assert k._nd == sample_data.shape[1]
    assert k._type == KERNEL_TYPE.NONPARAMETRIC

    # bandwidth not set yet – fallback path should be used later
    assert k.h is None
    # _ne should equal number of samples because no weights are supplied
    assert k._ne == k._ns


def test_instantiation_with_bandwidth(sample_data):
    """Providing a bandwidth list at construction must be stored correctly."""
    bw = np.array([0.5, 1.0])
    k = Cauchy(data=sample_data, h=bw, calculate_bw=False)

    # bandwidth should be stored unchanged
    assert np.allclose(k.h, bw)
    # univariate call should use the first element
    assert k.kf_univar(0.0) == 1.0  # u = 0 → kernel = 1


def test_univariate_kernel_values(sample_data):
    """Compare the implementation against the analytical Cauchy kernel."""
    bw = 2.0
    k = Cauchy(data=sample_data, h=bw, calculate_bw=False)

    # test a few points
    for u in np.linspace(-5, 5, 11):
        expected = 1.0 / (1.0 + (u / bw) ** 2)
        assert np.isclose(k.kf_univar(u), expected, atol=1e-12)


def test_univariate_missing_bandwidth_raises(sample_data):
    """Calling kf_univar without a bandwidth must raise a ValueError."""
    k = Cauchy(data=sample_data, calculate_bw=False)
    with pytest.raises(ValueError, match="Bandwidth `h` must be set."):
        k.kf_univar(1.0)


def test_multivariate_covariance_branch(sample_data):
    """When a covariance matrix is present and well‑conditioned, the multivariate
    kernel should use the analytic expression."""
    # Create a kernel with a known covariance matrix
    k = Cauchy(data=sample_data, calculate_bw=True)
    # Manually inject a positive‑definite covariance matrix
    cov = np.cov(sample_data, rowvar=False)
    k._cov = cov
    d = 2

    # pick a vector and compute the expected value
    z = np.array([0.3, -0.7])
    inv_cov = np.linalg.inv(cov)
    quad_form = z.T @ inv_cov @ z
    expected = gamma((d + 1) / 2.0) / (np.pi ** (d / 2.0) * gamma(0.5)) * (1.0 + quad_form) ** (-(d + 1) / 2.0)

    assert np.isclose(k.kf_multivar(z), expected, atol=1e-12)


def test_multivariate_fallback_branch(sample_data):
    """When _cov is None the fallback implementation should be used."""
    bw = [1.0, 2.0]
    k = Cauchy(data=sample_data, h=bw, calculate_bw=False)

    # ensure the covariance branch is disabled
    k._cov = None

    z = np.array([0.5, -1.0])
    h_arr = np.asarray(bw)
    scaled = z / h_arr

    denom = 1 + np.sum(scaled**2)
    d = 2
    # fallback formula from the source code
    expected = gamma((d + 1) / 2.0) / (np.pi ** (d / 2.0) * gamma(0.5)) * denom ** (-(d + 1) / 2.0)

    assert np.isclose(k.kf_multivar(z), expected, atol=1e-12)


def test_multivariate_missing_bandwidth_raises(sample_data):
    """If _cov is None and no bandwidth is supplied, a ValueError must be raised."""
    k = Cauchy(data=sample_data, calculate_bw=False)
    k._cov = None
    with pytest.raises(ValueError, match="Bandwidth `h` must be set for multivariate kernel."):
        k.kf_multivar(np.zeros(2))
