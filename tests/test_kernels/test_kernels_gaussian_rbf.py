"""
Unit‑tests for src/samplersLib/kernels.GaussianRBF

The tests cover:
1. Isotropic kernel – uses the minimum bandwidth from ``h``.
2. Anisotropic kernel – uses a supplied covariance matrix.
3. Fallback to isotropic when no covariance is present.
"""

import unittest

import numpy as np

from samplersLib.kernels import GaussianRBF


class TestGaussianRBF(unittest.TestCase):
    def setUp(self):
        self.u = np.array([1.0, 2.0])
        self.dummy_data = np.random.randn(5, 2)

    # ------------------------------------------------------------------
    # 1️⃣ Isotropic kernel – use provided h and disable auto‑bw
    # ------------------------------------------------------------------
    def test_isotropic_kernel_value(self):
        h_values = [0.5, 1.0]  # min = 0.5
        kernel = GaussianRBF(
            data=self.dummy_data,
            h=h_values,
            calculate_bw=False,  # keep the supplied bandwidths
        )
        norm_sq = np.linalg.norm(self.u) ** 2
        expected = np.exp(-norm_sq / (2 * 0.5**2))
        self.assertAlmostEqual(kernel.kf_multivar(self.u), expected, places=12)

    # ------------------------------------------------------------------
    # 2️⃣ Anisotropic kernel – same trick
    # ------------------------------------------------------------------
    def test_anisotropic_kernel_value(self):
        h_values = [1.0, 1.0]
        kernel = GaussianRBF(data=self.dummy_data, h=h_values, calculate_bw=False)
        cov = np.diag([2.0, 3.0])
        kernel._cov = cov
        kernel._inv_cov = np.linalg.inv(cov)
        inv_cov = np.linalg.inv(cov)
        quad_form = float(self.u @ inv_cov @ self.u)
        expected = np.exp(-0.5 * quad_form)
        self.assertAlmostEqual(kernel.kf_multivar(self.u), expected, places=12)

    # ------------------------------------------------------------------
    # 3️⃣ Fallback to isotropic when no covariance
    # ------------------------------------------------------------------
    def test_fallback_to_isotropic_when_no_cov(self):
        h_values = [0.8]
        kernel = GaussianRBF(data=self.dummy_data, h=h_values, calculate_bw=False)
        norm_sq = np.linalg.norm(self.u) ** 2
        expected = np.exp(-norm_sq / (2 * 0.8**2))
        self.assertAlmostEqual(kernel.kf_multivar(self.u), expected, places=12)


if __name__ == "__main__":
    unittest.main()
