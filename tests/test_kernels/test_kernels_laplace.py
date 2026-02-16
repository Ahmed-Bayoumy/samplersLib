import numpy as np
import pytest

from samplersLib.kernels import Laplace


def _univar_expected(u, h):
    """Reference implementation of the univariate Laplace kernel."""
    return (1.0 / (2 * h)) * np.exp(-np.abs(u) / h)


def _multivar_expected(u, h):
    """Reference implementation for multivariate product kernel."""
    h = np.atleast_1d(h)
    coeff = 1.0 / (2 * h)
    return np.prod(coeff * np.exp(-np.abs(u) / h), axis=-1)


@pytest.mark.parametrize(
    "h",
    [
        0.5,  # scalar bandwidth
        [0.3, 0.7],  # array bandwidth
    ],
)
def test_laplace_univariate(h):
    """Test the univariate Laplace kernel against a hand‑computed reference."""
    # Dummy data with correct dimensionality (2D) – only needed for Kernel init.
    dummy_data = np.array([[1, 2]])

    k = Laplace(
        data=dummy_data,
        h=h,
        calculate_bw=False,  # we provide the bandwidth ourselves
    )

    u_vals = np.linspace(-3, 3, 5)
    expected_h = h[0] if isinstance(h, (list, np.ndarray)) else h
    expected = _univar_expected(u_vals, expected_h)
    result = k.kf_univar(u_vals)

    assert np.allclose(result, expected, atol=1e-12), f"Univariate kernel failed for bandwidth {h}"


@pytest.mark.parametrize(
    "h",
    [
        0.5,
        [0.3, 0.7],
    ],
)
def test_laplace_multivariate(h):
    """Test the multivariate Laplace kernel against a hand‑computed reference."""
    dummy_data = np.array([[1, 2], [3, 4]])  # 2‑D data

    k = Laplace(
        data=dummy_data,
        h=h,
        calculate_bw=False,
    )

    u_vals = np.array(
        [
            [-0.5, 0.2],
            [1.0, -1.5],
            [0.0, 0.0],
        ]
    )
    expected = _multivar_expected(u_vals, h)
    result = k.kf_multivar(u_vals)

    assert np.allclose(result, expected, atol=1e-12), f"Multivariate kernel failed for bandwidth {h}"


if __name__ == "__main__":
    pytest.main([__file__])
