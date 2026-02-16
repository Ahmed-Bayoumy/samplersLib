import numpy as np
import pytest

from samplersLib.kernels import InverseMultiquadricRBF


@pytest.fixture
def sample_data():
    """Return a small non‑empty dataset for bandwidth estimation."""
    return np.array([[0.0, 0.0], [1.0, 2.0], [-1.5, 0.5]])


def test_isotropic_kernel_value(sample_data):
    """Check the isotropic IMQ kernel against the analytical formula."""
    h = [1.0]  # shape parameter
    k = InverseMultiquadricRBF(data=sample_data, h=h, calculate_bw=False)

    u = np.array([3.0, 4.0])  # ‖u‖ = 5
    expected = 1 / np.sqrt(5**2 + 1**2)  # 1/√26

    assert pytest.approx(k.kf_multivar(u), rel=1e-12) == expected


def test_anisotropic_kernel_value(sample_data):
    """Verify that the kernel uses Mahalanobis distance when _cov is set."""
    h = [0.5]
    k = InverseMultiquadricRBF(data=sample_data, h=h, calculate_bw=False)

    # Define a covariance matrix (positive‑definite)
    cov = np.array([[2.0, 0.3], [0.3, 1.0]])
    k._cov = cov

    u = np.array([1.0, 0.0])

    # Compute Mahalanobis distance manually
    inv_cov = np.linalg.inv(cov)
    mahal = np.sqrt(u.T @ inv_cov @ u)

    expected = 1.0 / np.sqrt(mahal**2 + 0.5**2)

    assert pytest.approx(k.kf_multivar(u), rel=1e-12) == expected


def test_bandwidth_minimum(sample_data):
    """When multiple h values are supplied, the minimum is used."""
    h = [3.0, 1.0]  # min is 1.0
    k = InverseMultiquadricRBF(data=sample_data, h=h, calculate_bw=False)

    u = np.array([2.0, 0.0])  # ‖u‖ = 2

    expected = 1 / np.sqrt(2**2 + 1.0**2)  # 1/√5
    assert pytest.approx(k.kf_multivar(u), rel=1e-12) == expected
