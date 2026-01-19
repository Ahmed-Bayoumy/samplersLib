import numpy as np
import pytest

from samplersLib.kernels import Sigmoid


@pytest.fixture
def kernel_isotropic():
    """Sigmoid kernel with a single isotropic bandwidth."""
    data = np.array([[0.0, 0.0], [1.0, 1.0]])  # 2‑D dummy data
    return Sigmoid(data=data, h=[0.5])  # one bandwidth for all dims


@pytest.fixture
def kernel_aniso():
    """Sigmoid kernel with an anisotropic covariance matrix."""
    data = np.array([[0.0, 0.0], [1.0, 2.0]])
    k = Sigmoid(data=data, h=[0.3, 0.7])
    k._cov = np.array([[2.0, 0.5], [0.5, 1.0]])  # SPD matrix
    return k


def test_multivariate_isotropic(kernel_isotropic):
    """Multivariate kernel should reduce to product of univariates."""
    u = np.array([0.5, -0.2])  # two‑dimensional input
    prod_univar = (2 / np.pi) * 1 / np.cosh(u[0]) * (2 / np.pi) * 1 / np.cosh(u[1])
    expected = prod_univar / np.prod(kernel_isotropic.h)

    assert np.allclose(kernel_isotropic.kf_multivar(u), expected, atol=1e-12)


def test_bounded_behavior(kernel_isotropic):
    """`bounded` currently returns the input unchanged."""
    u = np.array([0.01, 0.01])
    raw_val = (2 / np.pi) * 1 / np.cosh(u[0]) * (2 / np.pi) * 1 / np.cosh(u[1]) / np.prod(kernel_isotropic.h)
    bounded_val = kernel_isotropic.bounded(raw_val, np.prod(u))

    assert bounded_val == raw_val
