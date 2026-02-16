import numpy as np
import pytest

from samplersLib.kernels import Cosine


def _dummy_data():
    """Return a minimal non‑empty array required by the Kernel base class."""
    return np.array([[0.0]])  # shape (1,1)


def test_univariate_value_and_support():
    h = 1.0
    k = Cosine(data=_dummy_data(), h=[h])  # pass bandwidth as a list

    assert np.isclose(k.kf_univar(np.array([[0]])), 1 / (2 * h))
    expected_mid = (1 / (2 * h)) * np.cos(np.pi * 0.5)
    assert np.isclose(k.kf_univar(np.array([[h / 2]])), expected_mid)
    assert k.kf_univar(2 * h) == 0
    assert k.kf_univar(-2 * h) == 0


def test_multivariate_value_and_support():
    h = [1.0, 2.0]
    k = Cosine(data=np.array([[0.0, 0.0]]), h=h)  # bandwidth list matches dimensionality

    u = np.array([[0.5, 1.0]])
    prod_u = np.prod(u)
    prod_h = np.prod(h)
    expected = (1 / (2 * prod_h)) ** len(u) * np.cos(np.pi * prod_u / prod_h)
    assert np.isclose(k.kf_multivar(u), expected)

    u_out = np.array([[3.0, 1.0]])  # first coord outside support
    assert k.kf_multivar(u_out) == 0


def test_vectorized_input():
    h = 1.0
    k = Cosine(data=np.array([[0.0, 0.0, 0.0]]), h=[h])

    u_vec = np.array([0, 0.5, 1.2])
    out = k.kf_univar(u_vec)
    expected = (1 / (2 * h)) * np.cos(np.pi * u_vec / h)
    expected[np.abs(u_vec) > h] = 0
    assert np.allclose(out, expected)


if __name__ == "__main__":
    pytest.main()
