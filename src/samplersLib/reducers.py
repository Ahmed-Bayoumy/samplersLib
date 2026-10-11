# pylint: disable=too-many-lines
"""
# ------------------------------------------------------------------------------------#
#  Samplers Library - (SamplersLib)                                                   #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software          #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU General Public License for more details.          #
#                                                                                     #
#  You should have received a copy of the GNU General Public License along            #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on SamplersLib at                                         #
#  https://github.com/Ahmed-Bayoumy/samplersLib                                       #
#  Copyright (C) 2024-2026  Ahmed H. Bayoumy                                          #
# ------------------------------------------------------------------------------------#
"""

import copy

import numpy as np

#: Supported dimensionality-reduction methods for `Reducers`.
#: - "pca": eigendecomposition of the *correlation* matrix (data is both
#:   mean-centered and divided by its per-dimension std-dev before the
#:   decomposition), i.e. every original dimension is weighted equally
#:   regardless of scale. Standard choice, good default.
#: - "pod": classic "snapshot" Proper Orthogonal Decomposition (Sirovich,
#:   1987) - only mean-centers the data (no std-dev normalization), and
#:   extracts modes via direct SVD of the (mean-centered) snapshot matrix
#:   rather than eigendecomposition of the covariance matrix. Letting the
#:   raw (scaled) variance drive which directions are kept - instead of
#:   forcing every axis to unit variance - captures more of the
#:   directionally-relevant structure when the good/bad point cloud isn't
#:   evenly spread across dimensions, at the cost of being more sensitive
#:   to per-dimension scale than PCA's correlation-based decomposition.
METHODS = ("pca", "pod")


class Reducers:
    """
    Dimensional reduction of the search space
    """

    def __init__(
        self,
        data: np.ndarray,
        vlim: np.ndarray,
        nd: int,
        n_components: int = 3,
        seed: int = 12345,
        method: str = "pca",
    ):
        if method not in METHODS:
            raise ValueError(f"`method` must be one of {METHODS}, got {method!r}.")
        self.data = copy.deepcopy(data)
        self.var_limits = copy.deepcopy(vlim)
        self.n_d = nd
        self.n_components = n_components
        self.method = method
        # Compute the covariance matrix
        self.seed = seed
        self.standardize_data()
        self.data_standardized: np.ndarray = None

    def standardize_data(self):
        """
        Standardize the data points: always mean-centered against the
        min-max scaled data; additionally divided by the per-dimension
        std-dev when `method == "pca"` (POD skips that step - see `METHODS`
        above).
        """
        self.data_scaled = (self.data - self.var_limits[:, 0]) / (self.var_limits[:, 1] - self.var_limits[:, 0])
        self.means = np.mean(self.data_scaled, axis=0)
        if self.method == "pod":
            self.std_devs = np.ones_like(self.means)
        else:
            self.std_devs = np.std(self.data_scaled, axis=0)
        self.data_standardized = (self.data_scaled - self.means) / self.std_devs
        nan_indices = np.isnan(self.data_standardized)
        rng = np.random.default_rng(seed=self.seed)
        col_means = rng.normal(0, 1e-5, size=self.n_d)
        self.data_standardized[nan_indices] = np.take(col_means, np.where(nan_indices)[1])

    def rd(self):
        """
        Reduce the dimensional space to its leading modes: principal
        components (eigendecomposition of the covariance of the
        standardized data) for `method == "pca"`, or POD modes (SVD of the
        mean-centered snapshot matrix) for `method == "pod"`.
        """
        self.standardize_data()
        self.k = self.n_components  # Number of modes we want to reduce to

        if self.method == "pod":
            # Snapshot POD: modes are the leading right singular vectors of
            # the (mean-centered) snapshot matrix.
            _, _, vt = np.linalg.svd(self.data_standardized, full_matrices=False)
            self.eigenvectors_top_k = vt[: self.k, :].T
        else:
            cov_matrix = np.cov(self.data_standardized, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # Sort eigenvalues in descending order, and rearrange the eigenvectors accordingly
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvectors_sorted = eigenvectors[:, sorted_indices]
            self.eigenvectors_top_k = eigenvectors_sorted[:, : self.k]

        # Project the original data onto the new space
        return np.dot(self.data_standardized, self.eigenvectors_top_k)

    def project_rd_to_original_space(self, samples: np.ndarray):
        """
        Project reduced-space coordinates back to the original dimensional
        space.

        Inverts every step of `standardize_data`/`rd`: un-rotate via the
        PCA/POD basis, then undo the standardization (`* std_devs + means`
        - a no-op multiplication for `method == "pod"`, since `std_devs` is
        all ones there), then undo the min-max scaling against `var_limits`.
        """
        standardized = samples.dot(self.eigenvectors_top_k.T)
        scaled = standardized * self.std_devs + self.means
        return scaled * (self.var_limits[:, 1] - self.var_limits[:, 0]) + self.var_limits[:, 0]

    @classmethod
    def reduce(self, data: np.ndarray, vlim: np.ndarray, nd: int, n_components: int = 3, method: str = "pca"):
        red = Reducers(data=data, vlim=vlim, nd=nd, n_components=n_components, method=method)
        return red.rd()

    @classmethod
    def project(
        self,
        original_data: np.ndarray,
        samples: np.ndarray,
        vlim: np.ndarray,
        nd: int,
        n_components: int = 3,
        method: str = "pca",
    ):
        # `n_components`/`method` must match the values used to build
        # `samples` in the first place (i.e. the `reduce(...)` call this is
        # inverting) - otherwise the re-derived basis has the wrong shape,
        # or the wrong basis entirely, for `samples.dot(eigenvectors_top_k.T)`.
        red = Reducers(data=original_data, vlim=vlim, nd=nd, n_components=n_components, method=method)
        red.rd()
        return red.project_rd_to_original_space(samples=samples)


if __name__ == "__main__":
    """ Samplers library """
