import numpy as np
from scipy.integrate import dblquad, quad

from samplersLib.kernels import Triweight


def triweight_value(u, h=1.0):
    """Return K_h(u) for 1‑D or 2‑D (product form)."""
    dummy_data = np.zeros((1, 1))  # shape (n_samples, n_features)
    if np.ndim(u) == 0:
        k = Triweight(data=dummy_data, h=h, calculate_bw=False)
        return float(k.kf_univar(u))
    else:
        k = Triweight(data=dummy_data, h=[h] * len(u))
        return float(k.kf_multivar(np.array([u])))


def test_triweight_integral_1d():
    """∫ K(u) du over [-1, 1] ≈ 1."""
    f = lambda u: triweight_value(u)  # noqa: E731
    integral, _ = quad(f, -1.0, 1.0)
    assert np.isclose(integral, 1.0, atol=1e-4), f"Univariate integral {integral} differs from 1"


def test_triweight_product_2d():
    """Product kernel integrates to 1 over the 2‑D support."""
    dummy_data = np.zeros((1, 2))
    k = Triweight(data=dummy_data, h=[0.92899], calculate_bw=False)

    def f(u1, u2):
        return float(k.kf_multivar(np.array([u1, u2])))

    # Integrate over the support [-h_i, h_i] for each dimension
    integral, _ = dblquad(
        lambda u2, u1: f(u1, u2),
        -0.5,
        0.5,  # u1 limits (h1)
        lambda _: -1.0,
        lambda _: 1.0,  # u2 limits scaled by h2
    )
    assert np.isclose(integral, 1.0, atol=1e-2), f"Multivariate integral {integral} differs from 1"


if __name__ == "__main__":
    test_triweight_integral_1d()
    test_triweight_product_2d()
    print("All Triweight tests passed.")
