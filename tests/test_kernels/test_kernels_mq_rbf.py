import unittest

import numpy as np

from samplersLib.kernels import MultiquadricRBF


class TestMultiquadricRBF(unittest.TestCase):
    """Unit‑tests for the anisotropic / isotropic multiquadratic kernel."""

    def setUp(self) -> None:
        # Common parameters
        self.c = 0.7
        self.u = np.array([1.2, -0.5, 0.3])

        # Dummy data required by Kernel base class
        dummy_data = np.zeros((1, len(self.u)))  # shape (1,3)

        # Isotropic kernel (no covariance)
        self.iso_kernel = MultiquadricRBF(data=dummy_data, c=self.c)

        # Anisotropic kernel – use a simple diagonal covariance
        cov_diag = np.diag([2.0, 0.5, 1.0])  # Σ
        self.aniso_kernel = MultiquadricRBF(data=dummy_data, c=self.c)
        self.aniso_kernel._cov = cov_diag

    def test_isotropic_value(self):
        """Check that the isotropic kernel matches the analytic formula."""
        r2 = np.linalg.norm(self.u) ** 2
        expected = np.sqrt(r2 + self.c**2)

        actual = self.iso_kernel.kf_multivar(self.u)
        self.assertAlmostEqual(actual, expected, places=12, msg="Isotropic value mismatch")

    def test_anisotropic_value(self):
        """Check that the anisotropic kernel uses Mahalanobis distance."""
        inv_cov = np.linalg.inv(self.aniso_kernel._cov)
        r2 = self.u @ inv_cov @ self.u
        expected = np.sqrt(r2 + self.c**2)

        actual = self.aniso_kernel.kf_multivar(self.u)
        self.assertAlmostEqual(actual, expected, places=12, msg="Anisotropic value mismatch")

    def test_univar_delegates_to_multivar(self):
        """kf_univar should return the same as kf_multivar."""
        iso_val = self.iso_kernel.kf_univar(self.u)
        multivar_val = self.iso_kernel.kf_multivar(self.u)
        self.assertEqual(iso_val, multivar_val)

    def test_invalid_cov_shape_raises(self):
        """Providing a non‑square covariance should raise an error."""
        bad_cov = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2 but u is 3‑D
        kernel = MultiquadricRBF(data=np.zeros((1, len(self.u))), c=self.c)
        kernel._cov = bad_cov

        with self.assertRaises(ValueError):
            _ = kernel.kf_multivar(self.u)

    def test_zero_vector(self):
        """Kernel value for the zero vector should be simply c."""
        zero_vec = np.zeros_like(self.u)
        expected_iso = np.sqrt(0 + self.c**2)  # equals |c|
        actual_iso = self.iso_kernel.kf_multivar(zero_vec)

        expected_aniso = np.sqrt(0 + self.c**2)
        actual_aniso = self.aniso_kernel.kf_multivar(zero_vec)

        self.assertAlmostEqual(actual_iso, expected_iso, places=12)
        self.assertAlmostEqual(actual_aniso, expected_aniso, places=12)


if __name__ == "__main__":
    unittest.main()
