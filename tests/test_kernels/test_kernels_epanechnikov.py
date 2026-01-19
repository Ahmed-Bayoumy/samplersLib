import numpy as np
import pytest

from samplersLib.kernels import Epanechnikov


@pytest.fixture
def uni_kernel():
    """Return a univariate Epanechnikov kernel with bandwidth 1."""
    return Epanechnikov(data=np.array([[0]]), h=[1.0])


@pytest.fixture
def multi_kernel():
    """Return a multivariate Epanechnikov kernel in 2‑D with bandwidths [1, 1]."""
    return Epanechnikov(data=np.array([[0, 0]]), h=[1.0, 1.0])


def test_univar_support(uni_kernel):
    """Kernel should be zero outside its support."""
    assert uni_kernel.kf_univar(np.array([[1.5]])) == 0
    assert uni_kernel.kf_univar(np.array([[-2.0]])) == 0


def test_univar_peak(uni_kernel):
    """At u = 0 the kernel attains its maximum value."""
    peak = uni_kernel.kf_univar(np.array([0]))
    # For h=1, max should be 3/4
    assert pytest.approx(peak, rel=1e-6) == 0.75


def test_univar_symmetry(uni_kernel):
    """Kernel is symmetric around zero."""
    assert uni_kernel.kf_univar(0.5) == uni_kernel.kf_univar(np.array([-0.5]))


def test_multivar_support(multi_kernel):
    """Outside the unit ball kernel returns 0."""
    # Point outside radius sqrt(2)
    out = np.array([1.5, 1.5])
    assert multi_kernel.kf_multivar(out) == 0


def test_multivar_inside(multi_kernel):
    """Inside the support we get a positive value."""
    inside = np.array([0.3, -0.4])
    val = multi_kernel.kf_multivar(inside)
    # Should be > 0
    assert val > 0
    # Roughly check against analytic formula:
    h_prod = np.prod(multi_kernel.h)
    norm_sq = np.dot(inside, inside)
    expected = (3 / 4) * (1 - norm_sq / h_prod**2) / (h_prod ** len(inside))
    assert pytest.approx(val, rel=1e-6) == expected


def test_multivar_normalization(multi_kernel):
    """Numerical integration over the support should be close to 1."""
    # Monte‑Carlo estimate of integral
    rng = np.random.default_rng(seed=42)
    samples = rng.uniform(-1, 1, size=(100_000, 2))
    vals = np.array([multi_kernel.kf_multivar(u) for u in samples])
    volume = 2**2  # side length squared
    integral_estimate = vals.mean() * volume
    assert pytest.approx(integral_estimate, rel=0.18) == 1.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
