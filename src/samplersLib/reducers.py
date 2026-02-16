# pylint: disable=too-many-lines
"""
# ------------------------------------------------------------------------------------#
#  Samplers Library - (SamplersLib)                                                   #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the BSD 3-Clause License as published by the Free Software                #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the BSD 3-Clause License for more details.                #
#                                                                                     #
#  You should have received a copy of the BSD 3-Clause License along                  #
#  with this program. If not, see <https://opensource.org/license/bsd-3-clause/>.     #
#                                                                                     #
#  You can find information on SamplersLib at                                         #
#  https://github.com/Ahmed-Bayoumy/samplersLib                                       #
#  Copyright (C) 2024-2026  Ahmed H. Bayoumy                                          #
# ------------------------------------------------------------------------------------#
"""

import copy

import numpy as np


class Reducers:
    """
    Dimensional reduction of the search space
    """

    def __init__(self, data: np.ndarray, vlim: np.ndarray, nd: int, seed: int = 12345):
        self.data = copy.deepcopy(data)
        self.var_limits = copy.deepcopy(vlim)
        self.n_d = nd
        # Compute the covariance matrix
        self.seed = seed
        self.standardize_data()
        self.data_standardized: np.ndarray = None

    def standardize_data(self):
        """
        Standardize the data points
        """
        self.data_scaled = (self.data - self.var_limits[:, 0]) / (self.var_limits[:, 1] - self.var_limits[:, 0])
        self.means = np.mean(self.data_scaled, axis=0)
        self.std_devs = np.std(self.data_scaled, axis=0)
        self.data_standardized = (self.data_scaled - self.means) / self.std_devs
        nan_indices = np.isnan(self.data_standardized)
        rng = np.random.default_rng(seed=self.seed)
        col_means = rng.normal(0, 1e-5, size=self.n_d)
        self.data_standardized[nan_indices] = np.take(col_means, np.where(nan_indices)[1])

    def rd(self):
        """
        Reduce the dimensional space to principal components
        """
        # Compute the covariance matrix
        self.standardize_data()
        cov_matrix = np.cov(self.data_standardized, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues in descending order, and rearrange the eigenvectors accordingly
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]

        # Select the top 'k' eigenvectors to form the new matrix
        self.k = 3  # Number of principal components we want (reduce to 3D)
        self.eigenvectors_top_k = eigenvectors_sorted[:, : self.k]

        # Project the original data onto the new space
        return np.dot(self.data_standardized, self.eigenvectors_top_k)

    def project_rd_to_original_space(self, samples: np.ndarray):
        """
        Project principal components to the original dimensional space
        """
        return samples.dot((self.eigenvectors_top_k[:, : self.k].T) + self.means)

    @classmethod
    def reduce(self, data: np.ndarray, vlim: np.ndarray, nd: int):
        red = Reducers(data=data, vlim=vlim, nd=nd)
        return red.rd()

    @classmethod
    def project(self, original_data: np.ndarray, samples: np.ndarray, vlim: np.ndarray, nd: int):
        red = Reducers(data=original_data, vlim=vlim, nd=nd)
        red.rd()
        return red.project_rd_to_original_space(samples=samples)


if __name__ == "__main__":
    """ Samplers library """
