import numpy as np
import pytest

from samplersLib.kernels import UniformRectangular


def _indicator(limit: float, u: np.ndarray) -> float:
    """
    Helper that mimics the behaviour of `Kernel.bounded` used in the class.
    It returns 1.0 if |u| <= limit else 0.0.
    """
    return 1.0 if abs(u) <= limit else 0.0


@pytest.fixture
def isotropic_kernel():
    """Return a kernel with a simple 1‑D bandwidth and dummy data."""
    return UniformRectangular(data=np.array([[0.0]]), h=[1.0])


@pytest.fixture
def anisotropic_kernel():
    """Return a kernel that will use an explicit covariance matrix."""
    k = UniformRectangular(data=np.array([[0.0, 0.0]]), h=[1.0, 1.0])
    # Explicitly set a positive‑definite covariance matrix.
    k._cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    return k


class TestUniformRectangular:
    def test_isotropic_within_bounds(self, isotropic_kernel):
        """Kernel should be 1 inside the hyper‑rectangle."""
        u = np.array([0.3])  # |u| < 0.5
        result = isotropic_kernel.kf_multivar(u)
        assert result == pytest.approx(_indicator(0.5 / np.prod(isotropic_kernel.h), np.prod(u)))

    def test_isotropic_outside_bounds(self, isotropic_kernel):
        """Kernel should be 0 outside the hyper‑rectangle."""
        u = np.array([1.2])  # |u| > 0.5
        result = isotropic_kernel.kf_multivar(u)
        assert result == pytest.approx(_indicator(0.5 / np.prod(isotropic_kernel.h), np.prod(u)))

    def test_anisotropic_transformation(self, anisotropic_kernel):
        """
        Verify that the kernel uses the inverse Cholesky factor of the
        covariance matrix to transform the input before applying the
        indicator.
        """
        # Pick a vector that is inside the *whitened* unit cube but outside
        # the original axis‑aligned box.  For example, u = [1.0, 1.0]
        # will be transformed by inv(L) where L is the Cholesky of _cov.
        u_original = np.array([1.0, 1.0])

        # Compute the expected result manually
        L = np.linalg.cholesky(anisotropic_kernel._cov)
        invL = np.linalg.inv(L)
        u_transformed = invL @ u_original
        expected = _indicator(0.5, np.prod(u_transformed))

        result = anisotropic_kernel.kf_multivar(u_original)
        assert result == pytest.approx(expected)

    def test_anisotropic_fallback_to_isotropic(self, anisotropic_kernel):
        """
        If the covariance matrix is not positive‑definite,
        the kernel should fall back to the isotropic behaviour.
        """
        # Make _cov singular (non‑positive‑definite)
        anisotropic_kernel._cov = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank‑deficient

        u = np.array([0.3, 0.3])  # inside the isotropic box
        result = anisotropic_kernel.kf_multivar(u)
        expected = _indicator(0.5 / np.prod(anisotropic_kernel.h), np.prod(u))
        assert result == pytest.approx(expected)
