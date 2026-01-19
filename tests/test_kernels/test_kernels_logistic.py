import numpy as np
import pytest

from samplersLib.kernels import TUNING_METHOD, Logistic


@pytest.fixture
def sample_data():
    """Generate a small 2‑D dataset for bandwidth calculation."""
    return np.random.randn(50, 2)


def test_isotropic_bandwidth(sample_data):
    # Force a single scalar bandwidth
    k = Logistic(data=sample_data, h=[0.5], calculate_bw=False)
    u = np.array([1.0, -0.5])
    val = k.kf_multivar(u)
    denom_expected = 0.5 ** u.shape[-1]  # d = 2
    expected = np.prod(1 / (np.exp(u) + 2 + np.exp(-u))) / denom_expected
    assert pytest.approx(val, rel=1e-6) == expected


def test_anisotropic_no_cov(sample_data):
    k = Logistic(data=sample_data, h=[0.3, 0.7], calculate_bw=False)
    u = np.array([0.2, -1.1])
    val = k.kf_multivar(u)
    denom_expected = 0.3 * 0.7
    expected = np.prod(1 / (np.exp(u) + 2 + np.exp(-u))) / denom_expected
    assert pytest.approx(val, rel=1e-6) == expected


def test_anisotropic_with_cov(sample_data):
    # Create a kernel that will compute its own covariance
    k = Logistic(data=sample_data, bw_method=TUNING_METHOD.MLCV.name)
    # Manually set anisotropic bandwidths (will be scaled by cov later)
    k.h = [0.4, 0.6]
    # Inject a dummy covariance matrix (e.g., identity * variances)
    k._cov = np.array([[1.0, 0.0], [0.0, 4.0]])  # var_diag = [1, 2]

    u = np.array([0.5, -0.3])
    val = k.kf_multivar(u)

    var_diag = np.sqrt(np.diag(k._cov))
    scaled_h = np.array(k.h) * var_diag
    denom_expected = np.prod(scaled_h)
    expected = np.prod(1 / (np.exp(u) + 2 + np.exp(-u))) / denom_expected

    assert pytest.approx(val, rel=1e-6) == expected


def test_univar_kernel():
    dummy_data = np.array([[0.0]])
    k = Logistic(data=dummy_data)
    u = np.array([0.0, 1.5, -2.0])
    val = k.kf_univar(u)
    expected = 1 / (np.exp(u) + 2 + np.exp(-u))
    assert np.allclose(val, expected)
