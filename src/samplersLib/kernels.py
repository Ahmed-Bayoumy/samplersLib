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
#  Copyright (C) 2026  Ahmed H. Bayoumy                                               #
# ------------------------------------------------------------------------------------#
"""

import copy
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, List

import numpy as np
import plotly.express as px

from ._common import KERNEL_TYPE, TUNING_METHOD


class Kernel(ABC):
    """Base class for kernel functions"""

    _ns: int = 0
    _nd: int = 0
    data: np.ndarray = None
    x: np.ndarray = None
    h: List[float] = 0.1
    K: np.ndarray = None
    std_dev: float = None
    weights: Any = None
    _ne: int = 0
    _covariance_factor: Callable = None
    _factor: Any = None
    _data_inv_cov: np.ndarray = None
    _data_cov: np.ndarray = None
    _cov: np.ndarray = None
    _inv_cov: np.ndarray = None
    _log_det: Any = None
    res: int = 101
    _points: np.ndarray = None
    _point: np.ndarray = None
    est_pdf: np.ndarray = None
    vlim: np.ndarray = None
    bw_method: str = None
    is_debugging: bool = False
    _type: KERNEL_TYPE = KERNEL_TYPE.NONPARAMETRIC
    _calculate_bw: bool = True

    def __init__(
        self,
        data: np.ndarray = None,
        x: np.ndarray = None,
        h: List[float] = 0.1,
        std_dev: float = None,
        weights: Any = None,
        res: int = 101,
        est_pdf: np.ndarray = None,
        vlim: np.ndarray = None,
        bw_method: str = None,
        is_debugging: bool = False,
        calculate_bw=True,
    ):
        """Protocol class for kernel functions"""
        self._ns: int = 0
        self._nd: int = 0
        self.data: np.ndarray = data
        self.x: np.ndarray = x
        self.h: List[float] = h
        self.std_dev: float = std_dev
        self.weights: Any = weights
        self._ne: int = 0
        self._covariance_factor: Callable = None
        self._factor: Any = None
        self._data_inv_cov: np.ndarray = None
        self._data_cov: np.ndarray = None
        self._cov: np.ndarray = None
        self._inv_cov: np.ndarray = None
        self._log_det: Any = None
        self.res: int = res
        self._points: np.ndarray = None
        self._point: np.ndarray = None
        self.est_pdf: np.ndarray = est_pdf
        self.vlim: np.ndarray = vlim
        self.bw_method: str = bw_method
        self.is_debugging: bool = is_debugging
        self._type: KERNEL_TYPE = KERNEL_TYPE.NONPARAMETRIC
        self._calculate_bw = True

    @property
    def type(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        return self._type

    def _generate_nd_grid(self):
        """_summary_"""
        pt = []
        for i in range(self._nd):
            pt.append(np.linspace(self.vlim[i, 0], self.vlim[i, 1], int(self._ne)).T)

        self._points = np.vstack(np.meshgrid(*pt)).reshape(len(pt), -1).T
        self._points = self._points[: self._ne, :]

    def _calc_kf_per_dp(self, x, xi):
        """_summary_

        :param x: _description_
        :type x: _type_
        :param xi: _description_
        :type xi: _type_
        :return: _description_
        :rtype: _type_
        """
        diff = xi - x
        # Mahalanobis distance squared
        r2 = diff.T
        return self.kf_multivar(r2)

    def calculate(self, h: float = None):
        """_summary_

        :param h: _description_, defaults to None
        :type h: float, optional
        :return: _description_
        :rtype: _type_
        """
        bw = []
        if h is None:
            bw = self.h
        else:
            self.h = h
            bw = h

        x = np.asarray(self.data)
        for xi in self.data:
            xi = np.asarray(xi)
            bw.append(self._calc_kf_per_dp(x, xi))
        return sum(bw) / len(bw)

    def tune_bandwidth_mlcv(self, h_min=0.1, h_max=2.0, steps=20):
        """Finds the h that maximizes MLCV score via grid search."""
        best_h = h_min
        max_score = -float("inf")

        for i in range(steps):
            # Linear search across the range
            h = h_min + (h_max - h_min) * (i / (steps - 1))
            score = self.loo_log_likelihood(h)

            if score > max_score:
                max_score = score
                best_h = h

        self.h = best_h

    def tune_bw(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        if self.bw_method == TUNING_METHOD.MLCV.name:
            self.calculate()
        elif self.bw_method == TUNING_METHOD.SILVERMAN.name:
            self.bw_silverman()
        else:
            self.bw_scott()

        return self.h

    def set_bw(self, method):
        """_summary_

        :param method: _description_
        :type method: _type_
        """
        if method == TUNING_METHOD.MLCV.name:
            self._covariance_factor = self.tune_bandwidth_mlcv
        elif method == TUNING_METHOD.SILVERMAN.name:
            self._covariance_factor = self._silverman
        else:
            self._covariance_factor = self._scott

        self._calc_covariance()

    def _scott(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        return np.power(self._ns, -1.0 / (self._nd + 4))

    def _silverman(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        return np.power(self._ns * (self._nd + 2.0) / 4.0, -1.0 / (self._nd + 4))

    def bounded(self, f, u):
        """_summary_

        :param f: _description_
        :type f: _type_
        :param u: _description_
        :type u: _type_
        :return: _description_
        :rtype: _type_
        """
        return f if np.abs(u) < np.prod(self.h) else 0

    def _bw_calc(self, h):
        """_summary_

        :param h: _description_
        :type h: _type_
        :return: _description_
        :rtype: _type_
        """
        bandwidths = []
        stdev = []
        n = len(self.data)
        for j in range(self._nd):
            column = [row[j] for row in self.data]
            mean = sum(column) / n
            variance = sum((x - mean) ** 2 for x in column) / (n - 1)
            stdev.append(variance**0.5)
            bandwidths.append(h * stdev[-1])
        return bandwidths, stdev

    def bw_scott(self):
        """_summary_"""
        h = self._scott()
        self.h, self.std_dev = self._bw_calc(h)

    def bw_silverman(self):
        """_summary_"""
        h = self._silverman()
        self.h, self.std_dev = self._bw_calc(h)

    def calculate_std_devs(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        n, d = len(self.data), len(self.data[0])
        stds = []
        for j in range(d):
            col = [row[j] for row in self.data]
            mean = sum(col) / n
            var = sum((x - mean) ** 2 for x in col) / (n - 1)
            stds.append(np.sqrt(var))
        return stds

    def get_multivariate_bandwidths(self, method="mlcv"):
        """_summary_

        :param method: _description_, defaults to "mlcv"
        :type method: str, optional
        """
        self.std_dev = self.calculate_std_devs()

        # # 1. Standardize data so we can search for a single scalar 'h'
        # standardized_data = [
        #     [(row[j] / sigmas[j]) for j in range(d)]
        #     for row in self.data
        # ]

        if method == TUNING_METHOD.SCOTT:
            self.bw_scott()
        elif method == "silverman":
            self.bw_silverman()
        else:  # MLCV
            # Search for the h that works best in the standardized space
            self.bw_mlcv()

    def bw_mlcv(self):
        """_summary_"""
        self.tune_bandwidth_mlcv()

    def loo_log_likelihood(self, h):
        """Calculates the Leave-One-Out Log-Likelihood for a given bandwidth h."""
        n = len(self.data)
        total_log_lik = 0

        for i in range(n):
            # Leave-one-out estimate: sum of kernels for all j != i
            self.h = h
            self.set_bw(TUNING_METHOD.MLCV.name)
            xi = np.asarray(self.data[i])
            prob_i = sum(
                self._calc_kf_per_dp(self.data, xi) for j in range(n) if i != j
            ) / (n - 1)

            # Avoid log(0)
            total_log_lik += np.log(max(prob_i, 1e-15))

        return total_log_lik / n

    # TODO: Check if no longer needed
    # def _select_sigma(self, x):
    #     normalizer = 1.349
    #     iqr = (stats.scoreatpercentile(x, 75) - stats.scoreatpercentile(x, 25)) / normalizer
    #     std_dev = np.std(x, axis=0, ddof=1)
    #     return np.minimum(std_dev, iqr) if iqr > 0 else std_dev

    def _calc_covariance(self):
        """Computes the covariance matrix for each kernel using
        the kernel BW (covariance factor)."""
        try:
            self._factor = self._covariance_factor()

            if self._data_inv_cov is None:
                if self.weights is None:
                    self.weights = np.ones(self._ns) / self._ns
                self._data_cov = np.atleast_2d(
                    np.cov(self.data.T, rowvar=1, bias=True, aweights=self.weights)
                )
                self._data_inv_cov = np.linalg.inv(self._data_cov)

            self._cov = self._data_cov * self._factor**2
            self.is_debugging = False
            if self.is_debugging:
                labs = [f"x{i}" for i in range(self._nd)]
                # sns.heatmap(self._cov, annot=True, fmt='g', xticklabels=labs, yticklabels=labs)
                fig = px.imshow(self._cov, text_auto=True, x=labs, y=labs)
                fig.show()
                # plt.show()
            self._inv_cov = self._data_inv_cov / self._factor**2
            # L = np.linalg.cholesky(self._cov*2*np.pi)
            # self._log_det = 2*np.log(np.diag(L)).sum()
        except Exception:
            self._data_cov = None
            self._data_inv_cov = None
            self._cov = None
            self._inv_cov = None

    @abstractmethod
    # pylint: disable=missing-function-docstring
    def kf_univar(self):
        pass

    @abstractmethod
    # pylint: disable=missing-function-docstring
    def kf_multivar(self, r2):
        pass

    def estimate_pdf(self, points=None):
        """_summary_

        :param points: _description_, defaults to None
        :type points: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        if points is None:
            if self._points is None:
                self._generate_nd_grid()
            self._points = np.atleast_2d(np.asarray(self._points))
        else:
            self._points = points

        return (
            self.get_ke_non_param()
            if self._type == KERNEL_TYPE.NONPARAMETRIC
            else self.get_ke_param()
        )

    def get_ke_non_param(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        if not isinstance(self.h, list) or self.h is None:
            self.tune_bw()
        self.est_pdf = np.atleast_1d(np.zeros((self._points.shape[0])))
        for i in range(self._ns):
            ei = np.atleast_1d(np.zeros((self._points.shape[0])))
            for j in range(self._points.shape[0]):
                z: np.ndarray = self._points[j, :] - self.data[i, :]
                ei[j] = self.kf_multivar(z)
            self.est_pdf += ei / self._points.shape[0]

        self.est_pdf = np.atleast_1d(abs(self.est_pdf) / sum(abs(self.est_pdf)))

        return np.asarray(self.est_pdf)

    def get_ke_param(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        # if not isinstance(self.h, list) or self.h is None:
        #     self.tune_bw()
        self.est_pdf = np.atleast_1d(np.zeros((self._points.shape[0])))
        for i in range(self._ns):
            ei = np.atleast_1d(np.zeros((self._points.shape[0])))
            for j in range(self._points.shape[0]):
                z: np.ndarray = self._points[j, :] - self.data[i, :]
                ei[j] = self.kf_multivar(z)
            self.est_pdf += ei / self._points.shape[0]

        self.est_pdf = np.atleast_1d(abs(self.est_pdf) / sum(abs(self.est_pdf)))

        return np.asarray(self.est_pdf)

    def _sample_noise(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        h = np.array(self.h)
        kernel_name = self.__class__.__name__.lower()
        # 1. Infinite Support Kernels
        if kernel_name == "gaussian" or kernel_name == "gaussianrbf":
            # Box-Muller for Normal Distribution
            return [
                h
                * np.sqrt(-2 * np.log(random.random()))
                * np.cos(2 * np.pi * random.random())
                for _ in range(self._nd)
            ]

        elif kernel_name == "cauchy":
            # Inverse Transform Sampling: h * tan(pi * (U - 0.5))
            return [
                h * np.tan(np.pi * (random.random() - 0.5)) for _ in range(self._nd)
            ]

        elif kernel_name == "laplace":
            # Difference of two exponentials
            return [
                h * (np.log(random.random()) - np.log(random.random()))
                for _ in range(self._nd)
            ]

        elif kernel_name == "logistic":
            # h * log(U / (1-U))
            return [
                h * np.log(u / (1 - u)) if (u := random.random()) else 0
                for _ in range(self._nd)
            ]

        # 2. Compact Support Kernels (Bounded within [-h, h])
        # Using Rejection Sampling for complex shapes
        elif kernel_name in [
            "epanechnikov",
            "biweight",
            "triweight",
            "tricube",
            "cosine",
            "uniformrectangular",
        ]:
            noise = []
            for _ in range(self._nd):
                passed = False
                while True:
                    u, v = random.uniform(-1, 1), random.random()
                    if kernel_name == "epanechnikov":
                        passed = v <= (0.75 * (1 - u**2))
                    elif kernel_name == "biweight":
                        passed = v <= (15 / 16 * (1 - u**2) ** 2)
                    elif kernel_name == "triweight":
                        passed = v <= (35 / 32 * (1 - u**2) ** 3)
                    elif kernel_name == "tricube":
                        passed = v <= (70 / 81 * (1 - abs(u) ** 3) ** 3)
                    elif kernel_name == "cosine":
                        passed = v <= (np.pi / 4 * np.cos(np.pi / 2 * u))
                    elif kernel_name == "uniformrectangular":
                        passed = True
                    if passed:
                        noise.append(u * h)
                        break
            return noise

        # 3. RBF-Specific Sampling (Approximated for Generative Use)
        elif "multiquadricrbf" in kernel_name or "thinplatesplinerbf" in kernel_name:
            # RBFs aren't standard PDFs; they are sampled here via a high-variance Gaussian
            # to mimic the "influence" spread described in RBF theory.
            return [h * random.gauss(0, 2) for _ in range(self._nd)]

        return [0] * self._nd


class Gaussian(Kernel):
    """_summary_

    :param Kernel: _description_
    :type Kernel: _type_
    """

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method: str = "SCOTT",
        point: np.ndarray = None,
        n_r: int = 0,
        h: List[float] = None,
        calculate_bw=True,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        self.h = h
        self._calculate_bw = calculate_bw

        # Effective sample size
        if n_r > 0:
            self._ne = n_r
        # elif self.weights is not None:
        #     self._ne = int(1 / np.sum(self.weights ** 2))
        else:
            self._ne = self._ns

        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.NONPARAMETRIC
        if self._cov is None and self._calculate_bw:
            self.set_bw(method=bw_method)

    def kf_univar(self, u):
        if self.h is None:
            raise ValueError("Bandwidth `h` must be set.")
        h = self.h[0] if isinstance(self.h, (list, np.ndarray)) else self.h
        return (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(-0.5 * (u / h) ** 2)

    def kf_multivar(self, z: np.ndarray):
        if self._cov is not None:
            det = np.linalg.det(self._cov)
            if det > 1e-8:
                inv_cov = np.linalg.inv(self._cov)
                norm_const = 1.0 / np.sqrt((2 * np.pi) ** self._nd * det)
                exponent = -0.5 * z.T @ inv_cov @ z
                return norm_const * np.exp(exponent)

        # Fallback: Diagonal bandwidth assumption
        if self.h is None:
            raise ValueError(
                "Bandwidth `h` must be set for fallback multivariate kernel."
            )

        h_arr = np.asarray(self.h)
        scaled_z = z / h_arr
        norm_const = np.prod(1 / (np.sqrt(2 * np.pi) * h_arr))
        return norm_const * np.exp(-0.5 * np.sum(scaled_z**2))


class Cauchy(Kernel):
    """_summary_

    :param Kernel: _description_
    :type Kernel: _type_
    """

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method: str = "SCOTT",
        point: np.ndarray = None,
        n_r: int = 0,
        h: List[float] = None,
        calculate_bw: bool = True,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        self.h = h
        self._calculate_bw = calculate_bw
        if self._cov is None and self._calculate_bw:
            self.set_bw(method=bw_method)

        if n_r > 0:
            self._ne = n_r
        # elif self.weights is not None:
        #     self._ne = 1 / np.sum(self.weights ** 2)
        else:
            self._ne = self._ns

        self.vlim = np.atleast_2d(vlim)
        self._type = KERNEL_TYPE.NONPARAMETRIC

    def kf_univar(self, u):
        if self.h is None:
            raise ValueError("Bandwidth `h` must be set.")
        h = self.h[0] if isinstance(self.h, (list, np.ndarray)) else self.h
        return 1.0 / (1.0 + (u / h) ** 2)

    def kf_multivar(self, z: np.ndarray):
        if self._cov is not None:
            det = np.linalg.det(self._cov)
            if det > 1e-8:
                inv_cov = np.linalg.inv(self._cov)
                quad_form = z.T @ inv_cov @ z
                return 1.0 / ((1.0 + quad_form) ** ((self._nd + 1) / 2.0))

        if self.h is None:
            raise ValueError(
                "Bandwidth `h` must be set for fallback multivariate kernel."
            )

        h_arr = np.asarray(self.h)
        scaled_z = z / h_arr
        quad_form = np.sum(scaled_z**2)
        return (1 / (np.pi * (1 + scaled_z**2))).mean()


class Epanechnikov(Kernel):
    """_summary_

    :param Kernel: _description_
    :type Kernel: _type_
    """

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.MLCV.name,
        point: np.ndarray = None,
        h: List[float] = None,
        calculate_bw: bool = True,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        self._ne = self._ns
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.NONPARAMETRIC
        self.h = h
        self._calculate_bw = calculate_bw
        if self._cov is None and self._calculate_bw:
            self.set_bw(method=bw_method)

    """ Epanechnikov kernel function """

    def kf_univar(self, u):
        # return self.bounded((3 / 4 * (1 - u * u)), u)
        h = (
            self.h[0]
            if (isinstance(self.h, list) or isinstance(self.h, np.ndarray))
            and len(self.h) == 1
            else self.h
        )
        return (3 / (4 * h)) * (1 - ((u) / h) ** 2) * (np.abs(u) < h)

    def kf_multivar(self, u):
        # return self.bounded(np.prod(3 / 4 * (1 - u * u))/np.prod(self.h), np.prod(u))
        h = np.prod(self.h)
        d = u.shape[0]  # Dimension of the input
        norm_squared = np.dot(u, u)

        if norm_squared <= h**2:
            return (d / 2) * (1 - norm_squared / h**2)
        else:
            return 0


class Laplace(Kernel):
    """_summary_

    :param Kernel: _description_
    :type Kernel: _type_
    """

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method: str = "MLCV",
        point: np.ndarray = None,
        h: List[float] = None,
        calculate_bw: bool = True,
    ):
        if data is not None:
            self.data = np.atleast_2d(copy.deepcopy(data))
            self._ns = self.data.shape[0]  # num samples
            self._nd = self.data.shape[1]  # num dimensions
        else:
            raise ValueError("Data must be provided.")

        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / np.sum(np.square(self.weights))
        else:
            self._ne = self._ns

        self.vlim = np.atleast_2d(vlim) if vlim is not None else None
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.NONPARAMETRIC
        self.h = (
            np.array(h) if h is not None else np.ones(self._nd)
        )  # fallback bandwidth
        self._calculate_bw = calculate_bw
        if self._cov is None and self._calculate_bw:
            self.set_bw(method=bw_method)

    def kf_univar(self, u: np.ndarray) -> np.ndarray:
        """Univariate Laplace kernel function."""
        h = self.h[0] if isinstance(self.h, (list, np.ndarray)) else self.h
        return (1 / (2 * h)) * np.exp(-np.abs(u) / h)

    def kf_multivar(self, u: np.ndarray) -> np.ndarray:
        """Multivariate Laplace kernel function (product of univariate Laplace)."""
        h = np.asarray(self.h)
        coeff = 1 / (2 * h)
        return np.prod(coeff * np.exp(-np.abs(u) / h), axis=-1)


class Cosine(Kernel):
    """Cosine kernel function"""

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.MLCV.name,
        point: np.ndarray = None,
        h: List[float] = None,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        self._ne = self._ns
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.NONPARAMETRIC
        self.h = h

    def kf_univar(self, u):
        h = (
            self.h[0]
            if (isinstance(self.h, list) or isinstance(self.h, np.ndarray))
            and len(self.h) == 1
            else self.h
        )
        return self.bounded((1 / (2 * h)) * np.cos(np.pi * u / h), u=u)

    def kf_multivar(self, u):
        return self.bounded(
            (1 / (2 * np.prod(self.h)) ** len(u))
            * np.cos(np.pi * np.prod(u) / np.prod(self.h)),
            np.prod(u),
        )


class Linear(Kernel):
    """Linear  kernel function"""

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.MLCV.name,
        point: np.ndarray = None,
        h: List[float] = None,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        self._ne = self._ns
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.NONPARAMETRIC
        self.h = h

    def kf_univar(self, u):
        h = (
            self.h[0]
            if (isinstance(self.h, list) or isinstance(self.h, np.ndarray))
            and len(self.h) == 1
            else self.h
        )
        return max(0, 1 - abs(u) / h)

    def kf_multivar(self, u):
        return self.bounded(np.dot(u, u), np.linalg.norm(u))


class UniformRectangular(Kernel):
    """Uniform rectangular kernel function"""

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.MLCV.name,
        point: np.ndarray = None,
        h: List[float] = None,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        self._ne = self._ns
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.NONPARAMETRIC
        self.h = h

    def kf_univar(self, u):
        return self.bounded(0.5, u)

    def kf_multivar(self, u):
        return self.bounded(0.5 / np.prod(self.h), np.prod(u))


class Triweight(Kernel):
    """Triweight kernel function"""

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.MLCV.name,
        point: np.ndarray = None,
        h: List[float] = None,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        self._ne = self._ns
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.NONPARAMETRIC
        self.h = h

    def kf_univar(self, u):
        return self.bounded((35 / 32) * (1 - u**2) ** 3, u)

    def kf_multivar(self, u):
        return self.bounded(
            np.prod((35 / 32) * (1 - u**2) ** 3) / np.prod(self.h), np.prod(u)
        )


class Tricube(Kernel):
    """tricube kernel function"""

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.MLCV.name,
        point: np.ndarray = None,
        h: List[float] = None,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        self._ne = self._ns
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.NONPARAMETRIC
        self.h = h

    def kf_univar(self, u):
        return self.bounded((70 / 81) * (1 - np.abs(u) ** 3) ** 3, u)

    def kf_multivar(self, u):
        return self.bounded(
            np.prod((70 / 81) * (1 - np.abs(u) ** 3) ** 3) / np.prod(self.h), np.prod(u)
        )


class Silverman(Kernel):
    """Silverman kernel function"""

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.MLCV.name,
        point: np.ndarray = None,
        h: List[float] = None,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        self._ne = self._ns
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.NONPARAMETRIC
        self.h = h

    def kf_univar(self, u):
        return (
            0.5
            * np.exp(-(abs(u)) / np.sqrt(2))
            * np.sin((abs(u) / np.sqrt(2)) + (np.pi / 4))
        )

    def kf_multivar(self, u):
        h = np.array(self.h)
        u = np.array(u)
        return self.bounded(
            np.prod(
                0.5
                * np.exp(-(abs(u)) / np.sqrt(2))
                * np.sin((abs(u) / np.sqrt(2)) + (np.pi / 4))
            )
            / np.prod(h),
            np.prod(u),
        )


class Sigmoid(Kernel):
    """Sigmoid kernel function"""

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.MLCV.name,
        point: np.ndarray = None,
        h: List[float] = None,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        self._ne = self._ns
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.NONPARAMETRIC
        self.h = h

    def kf_univar(self, u):
        return (2 / np.pi) * (1 / (np.exp(u) + np.exp(-u)))

    def kf_multivar(self, u):
        return self.bounded(
            np.prod((2 / np.pi) * (1 / (np.exp(u) + np.exp(-u)))) / np.prod(self.h),
            np.prod(u),
        )


class Biweight(Kernel):
    """biweight kernel function"""

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.MLCV.name,
        point: np.ndarray = None,
        h: List[float] = None,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        self._ne = self._ns
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.NONPARAMETRIC
        self.h = h

    def kf_univar(self, u):
        return self.bounded((15 / 16) * (1 - u**2) ** 2, u)

    def kf_multivar(self, u):
        return self.bounded(
            np.prod((15 / 16) * (1 - u**2) ** 2) / np.prod(self.h), np.prod(u)
        )


class Logistic(Kernel):
    """logistic kernel function"""

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.MLCV.name,
        point: np.ndarray = None,
        h: List[float] = None,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        self._ne = self._ns
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.NONPARAMETRIC
        self.h = h

    def kf_univar(self, u):
        return 1 / (np.exp(u) + 2 + np.exp(-u))

    def kf_multivar(self, u):
        return np.prod(1 / (np.exp(u) + 2 + np.exp(-u))) / np.prod(self.h)


class GaussianRBF(Kernel):
    """_summary_

    :param Kernel: _description_
    :type Kernel: _type_
    """

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.SCOTT.name,
        point: np.ndarray = None,
        n_r: int = 0,
        h: List[float] = None,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        if n_r > 0:
            self._ne = n_r
        else:
            self._ne = self._ns
        if bw_method != TUNING_METHOD.MLCV.name:
            self.set_bw(bw_method)
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.PARAMETRIC
        self.h = h

    def kf_univar(self, u):
        return self.kf_multivar(u)

    def kf_multivar(self, u):
        h = np.array(self.h)
        return np.exp(-(np.linalg.norm(u) ** 2) / (2 * min(h) ** 2))


class MultiquadricRBF(Kernel):
    """_summary_

    :param Kernel: _description_
    :type Kernel: _type_
    """

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.SCOTT.name,
        point: np.ndarray = None,
        n_r: int = 0,
        h: List[float] = None,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        if n_r > 0:
            self._ne = n_r
        else:
            self._ne = self._ns
        if bw_method != TUNING_METHOD.MLCV.name:
            self.set_bw(bw_method)
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.PARAMETRIC
        self.h = h

    def kf_univar(self, u):
        return self.kf_multivar(u)

    def kf_multivar(self, u):
        return np.exp(-(np.linalg.norm(u) ** 2) / (2 * min(self.h) ** 2))


class InverseMultiquadricRBF(Kernel):
    """_summary_

    :param Kernel: _description_
    :type Kernel: _type_
    """

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.SCOTT.name,
        point: np.ndarray = None,
        n_r: int = 0,
        h: List[float] = None,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        if n_r > 0:
            self._ne = n_r
        else:
            self._ne = self._ns
        if bw_method != TUNING_METHOD.MLCV.name:
            self.set_bw(bw_method)
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.PARAMETRIC
        self.h = h

    def kf_univar(self, u):
        return self.kf_multivar(u)

    def kf_multivar(self, u):
        return 1 / np.sqrt(np.linalg.norm(u) ** 2 + min(self.h) ** 2)


class ThinPlateSplineRBF(Kernel):
    """_summary_

    :param Kernel: _description_
    :type Kernel: _type_
    """

    def __init__(
        self,
        data: np.ndarray = None,
        vlim=None,
        res: int = 101,
        weights: Any = None,
        bw_method=TUNING_METHOD.SCOTT.name,
        point: np.ndarray = None,
        n_r: int = 0,
        h: List[float] = None,
    ):
        self.data = copy.deepcopy(data)
        self._ns = data.shape[0]
        self._nd = data.shape[1]
        self.weights = weights
        if self.weights is not None:
            self._ne = 1 / sum(self.weights**2)
        if n_r > 0:
            self._ne = n_r
        else:
            self._ne = self._ns
        if bw_method != TUNING_METHOD.MLCV.name:
            self.set_bw(bw_method)
        self.vlim = np.atleast_2d(vlim)
        self.bw_method = bw_method
        self._type = KERNEL_TYPE.PARAMETRIC
        self.h = h

    def kf_univar(self, u):
        return self.kf_multivar(u)

    def kf_multivar(self, u):
        r = np.linalg.norm(u)
        return r**2 * np.log(r) if r > 0 else 0
