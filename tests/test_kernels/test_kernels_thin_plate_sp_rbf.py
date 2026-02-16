import numpy as np
import pytest

from samplersLib.kernels import ThinPlateSplineRBF


@pytest.mark.parametrize(
    "u, expected",
    [
        (0.0, 0.0),
        (1.0, 1.0 * np.log(1.0)),
        (-2.5, 6.25 * np.log(2.5)),
        (np.array([3.0]), 9.0 * np.log(3.0)),  # array input
    ],
)
def test_kf_univar(u, expected):
    """Test the univariate kernel for scalar and 1‑D array inputs."""
    k = ThinPlateSplineRBF(data=np.array([[0, 0]]))  # dummy data
    assert pytest.approx(k.kf_univar(u)) == expected


@pytest.mark.parametrize(
    "u, cov, expected",
    [
        # Isotropic case – Euclidean norm
        (np.array([3.0, 4.0]), None, 25.0 * np.log(5.0)),
        # Zero vector → zero kernel value
        (np.zeros(2), None, 0.0),
        # Anisotropic case – Mahalanobis distance with a diagonal covariance
        (
            np.array([1.0, 2.0]),
            np.diag([4.0, 9.0]),  # Σ = diag(4,9)
            (np.sqrt((1.0**2) / 4 + (2.0**2) / 9)) ** 2 * np.log(np.sqrt((1.0**2) / 4 + (2.0**2) / 9)),
        ),
    ],
)
def test_kf_multivar(u, cov, expected):
    """Test the multivariate kernel for isotropic and anisotropic cases."""
    k = ThinPlateSplineRBF(data=np.array([[0, 0]]))  # dummy data
    if cov is not None:
        k._cov = cov  # set covariance manually
        k._inv_cov = None  # force recomputation

    assert pytest.approx(k.kf_multivar(u)) == expected


def test_kf_multivar_caching():
    """Ensure that the inverse covariance matrix is cached after first use."""
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    k = ThinPlateSplineRBF(data=np.array([[0, 0]]))  # dummy data
    k._cov = cov
    k._inv_cov = None

    _ = k.kf_multivar(np.array([1.0, -1.0]))
    assert k._inv_cov is not None  # cached now

    inv_before = k._inv_cov.copy()

    _ = k.kf_multivar(np.array([-2.0, 3.0]))
    np.testing.assert_array_equal(k._inv_cov, inv_before)
