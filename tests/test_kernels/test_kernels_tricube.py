import numpy as np
import pytest

from samplersLib.kernels import Tricube


def _make_kernel(data=None, h_override=None):
    """
    Create a Tricube instance with minimal valid data.
    If `data` is None, generate a 10×d array of random numbers.
    Optionally override the bandwidth vector after construction.
    """
    if data is None:
        # Default to 10 samples in 3 dimensions
        data = np.random.randn(10, 3)

    kernel = Tricube(data=data)  # Kernel will validate `data`

    # Override bandwidth vector if requested
    if h_override is not None:
        kernel.h = np.array(h_override)

    return kernel


def test_univariate_support():
    """Kernel should be zero outside its support [-1, 1]."""
    k = _make_kernel(h_override=[1.0])
    assert k.kf_univar(2.0) == 0.0
    assert k.kf_univar(-3.5) == 0.0

    u = 0.5
    expected = (70 / 81) * (1 - abs(u) ** 3) ** 3
    assert pytest.approx(k.kf_univar(u), rel=1e-6) == expected


def test_multivariate_isotropic():
    """With a scalar bandwidth, the multivariate kernel reduces to the product of univariates."""
    h_scalar = 0.5
    k = _make_kernel(h_override=[h_scalar])  # force isotropic bandwidth

    u = np.array([0.1, -0.2, 0.05])
    prod_univar = np.prod([(70 / 81) * (1 - abs(ui) ** 3) ** 3 for ui in u])

    expected = prod_univar / np.prod(h_scalar)
    assert pytest.approx(k.kf_multivar(u), rel=1e-6) == expected


def test_multivariate_anisotropic():
    """When a bandwidth vector is supplied, each dimension gets its own bandwidth."""
    h_vector = np.array([0.8, 1.2])  # arbitrary anisotropic bandwidths
    k = _make_kernel(h_override=h_vector)

    u = np.array([0.2, -0.3])
    prod_univar = np.prod([(70 / 81) * (1 - abs(ui) ** 3) ** 3 for ui in u])

    expected = prod_univar / np.prod(h_vector)
    assert pytest.approx(k.kf_multivar(u), rel=1e-6) == expected


if __name__ == "__main__":
    pytest.main([__file__])
