import numpy as np
import pytest

from samplersLib.kernels import Linear


@pytest.fixture
def linear_scalar():
    """Kernel with a scalar bandwidth."""
    return Linear(data=np.array([[0.0]]), h=[0.5])  # h will be treated as 0.5


@pytest.fixture
def linear_vector():
    """Kernel with a vector bandwidth (length‑1 array)."""
    return Linear(data=np.array([[0.0]]), h=np.array([1.0]))


def test_kf_univar_scalar(linear_scalar):
    """Univariate kernel returns correct values for scalar bandwidth."""
    h = 0.5
    # Inside support
    assert linear_scalar.kf_univar(0) == pytest.approx(1.0)
    assert linear_scalar.kf_univar(h / 2) == pytest.approx(0.5)
    # At boundary
    assert linear_scalar.kf_univar(h) == pytest.approx(0.0)
    # Outside support
    assert linear_scalar.kf_univar(h * 1.5) == pytest.approx(0.0)


def test_kf_univar_vector(linear_vector):
    """Univariate kernel works the same when bandwidth is a vector."""
    h = 1.0
    assert linear_vector.kf_univar(0) == pytest.approx(1.0)
    assert linear_vector.kf_univar(h / 2) == pytest.approx(0.5)
    assert linear_vector.kf_univar(h) == pytest.approx(0.0)
    assert linear_vector.kf_univar(h * 2) == pytest.approx(0.0)


def test_kf_multivar_scalar(linear_scalar):
    """Multivariate kernel returns correct values for scalar bandwidth."""
    h = 0.5
    # Zero vector -> weight 1
    assert linear_scalar.kf_multivar(np.zeros(3)) == pytest.approx(1.0)
    # Inside support
    vec = np.array([h / 2, 0, 0])
    expected = max(0, 1 - np.linalg.norm(vec) / h)
    assert linear_scalar.kf_multivar(vec) == pytest.approx(expected)
    # At boundary
    vec = np.array([h, 0, 0])
    assert linear_scalar.kf_multivar(vec) == pytest.approx(0.0)
    # Outside support
    vec = np.array([h * 1.5, 0, 0])
    assert linear_scalar.kf_multivar(vec) == pytest.approx(0.0)


def test_kf_multivar_vector(linear_vector):
    """Multivariate kernel works with vector bandwidth."""
    h = 1.0
    vec = np.array([h / 2, h / 2, 0])
    expected = max(0, 1 - np.linalg.norm(vec) / h)
    assert linear_vector.kf_multivar(vec) == pytest.approx(expected)


def test_bounded_behavior(linear_scalar):
    """Ensure that the kernel never returns a value < 0 or > 1."""
    # Negative distance (should be clipped to 0)
    assert linear_scalar.kf_univar(-10) == pytest.approx(0.0)

    # Very large positive distance
    assert linear_scalar.kf_univar(1000) == pytest.approx(0.0)

    # Multivariate with huge norm
    vec = np.array([100, 200])
    assert linear_scalar.kf_multivar(vec) == pytest.approx(0.0)


if __name__ == "__main__":
    pytest.main()
