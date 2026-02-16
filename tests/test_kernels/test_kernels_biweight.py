import unittest

import numpy as np

from samplersLib.kernels import Biweight


class TestBiweightKernel(unittest.TestCase):
    """Unit‑tests for :class:`~samplersLib.kernels.Biwidth`."""

    def setUp(self) -> None:
        self.data = np.random.randn(100, 2)

    # ------------------------------------------------------------------
    # Helper: create a Biweight instance with optional covariance
    # ------------------------------------------------------------------
    def _make_kernel(self, cov=None):
        k = Biweight(data=self.data, bw_method="mlcv", h=[0.5, 0.5], calculate_bw=False)
        if cov is not None:
            k._cov = np.array(cov)  # expose private attribute for test
        return k

    # ------------------------------------------------------------------
    # Isotropic tests
    # ------------------------------------------------------------------
    def test_isotropic_univar(self):
        """Univariate kernel returns expected value at u=0."""
        k = self._make_kernel()
        val = k.kf_univar(0.0)
        self.assertAlmostEqual(val, 15 / 16)

    def test_isotropic_multivar(self):
        """Multivariate kernel matches product formula when no covariance."""
        k = self._make_kernel()
        u = np.array([0.2, -0.1])  # inside support
        prod_term = np.prod((15 / 16) * (1 - u**2) ** 2)
        expected = prod_term / np.prod(k.h)
        val = k.kf_multivar(u)
        self.assertAlmostEqual(val, expected)

    # ------------------------------------------------------------------
    # Anisotropic tests
    # ------------------------------------------------------------------
    def test_anisotropic_kernel(self):
        """Kernel uses covariance determinant when _cov is set."""
        cov = [[2.0, 0.5], [0.5, 1.0]]
        k = self._make_kernel(cov=cov)
        u = np.array([0.15, -0.05])  # inside support
        d = len(u)
        det_sqrt = np.sqrt(np.linalg.det(cov))
        expected = ((15 / 16) ** d) * np.prod((1 - u**2) ** 2) / det_sqrt
        val = k.kf_multivar(u)
        self.assertAlmostEqual(val, expected)

    def test_anisotropic_bounded(self):
        """Values outside the support are clipped to zero."""
        cov = [[1.0, 0], [0, 1.0]]
        k = self._make_kernel(cov=cov)
        u = np.array([2.0, 0.0])  # |u| > 1 → should be 0
        val = k.kf_multivar(u)
        self.assertEqual(val, 0.0)


if __name__ == "__main__":
    unittest.main()
