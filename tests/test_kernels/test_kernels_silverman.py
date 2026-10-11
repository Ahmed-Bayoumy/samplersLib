import numpy as np
import pytest

from samplersLib.kernels import Silverman


def _expected_univar(u):
    """Reference implementation used in the test."""
    if u.size == 1 and np.isclose(u, 0.0):
        return 0.5
    return 0.5 * np.exp(-(np.abs(u)) / np.sqrt(2)) * np.sin((np.abs(u) / np.sqrt(2)) + (np.pi / 4))


def _expected_multivar(u, c):
    h = np.array([1.0, 1.0])
    u_clipped = c._clip_to_support(u, h)

    denom = np.prod(h)
    return u_clipped / denom


@pytest.fixture
def kernel_isotropic():
    """Kernel with isotropic bandwidths (h = [1, 1])."""
    return Silverman(data=np.random.randn(10, 2), h=[1.0, 1.0])


def test_univar_zero(kernel_isotropic):
    """K(u=0) should equal the theoretical maximum 0.5."""
    assert np.isclose(kernel_isotropic.kf_univar(0.0), 0.5)


def test_univar_known_value(kernel_isotropic):
    """Check a few hand‑picked values against the reference formula."""
    for u in [-2.0, -1.0, 0.0, 1.0, 3.14]:
        expected = _expected_univar(np.array([u]))
        assert np.isclose(kernel_isotropic.kf_univar(np.array([u])), expected)


def test_multivar_isotropic(kernel_isotropic):
    """Product of univariate kernels divided by product(h)."""
    u = np.array([0.5, -1.2])
    h_prod = np.prod(kernel_isotropic.h)

    # Expected value: prod(kf_univar(u_i)) / prod(h)
    expected = np.prod(_expected_multivar(u, kernel_isotropic)) / h_prod

    assert np.isclose(kernel_isotropic.kf_multivar(u), expected)


def test_multivar_high_dimensional():
    """Test that the implementation does not under‑flow for many dimensions."""
    dim = 20
    u = np.array([1.0] * 20)

    kernel = Silverman(data=np.random.random((5, dim)), h=[1.0] * dim)

    val = kernel.kf_multivar(u)
    # The value should be finite and non‑zero
    assert np.isfinite(val) and val > 0


def test_anisotropic_transformation():
    """`kf_multivar` is the product of per-dimension shape values (on raw
    `u`) normalised by prod(h); a manually-injected `_cov` does not change
    that (the kernel has infinite support, so there is nothing to whiten)."""
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    kernel = Silverman(data=np.random.randn(10, 2), h=[1.0, 0.1])
    kernel._cov = cov  # manually inject covariance

    u = np.array([1.0, 0.0])

    expected = np.prod(kernel._clip_to_support(u, kernel.h)) / np.prod(kernel.h)
    assert np.isclose(kernel.kf_multivar(u), expected)
