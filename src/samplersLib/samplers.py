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
from abc import ABC, abstractmethod  # pylint: disable=too-many-lines

import math
from typing import Callable, Dict, Any, List
import copy
from dataclasses import dataclass
import random
from pyDOE2 import lhs
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import norm
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from .kernels import Kernel as KERNEL, Gaussian, Epanechnikov, Cosine, Linear, UniformRectangular, Triweight, Tricube, \
    Silverman, Sigmoid, Biweight, Logistic, Laplace, \
    GaussianRBF, MultiquadricRBF, InverseMultiquadricRBF, \
    ThinPlateSplineRBF, Cauchy
from ._common import *
from .particle import Particle
from .importance import RandomForest as RF
from .metrics import TrustWorthiness, compute_dimension_relevance
from .predictors import AdaptiveEnsemble, KNNKernelWeighted, MDNInspired, \
    GPWithMixedKernel, KernelWeightedAverage, LocalPolynomialRegression, \
    KernelRidgeRegression

from .reducers import Reducers


class Sampling(ABC):
    """
    An abstract base class representing various sampling methods.
    This class defines the interface for different sampling techniques, providing a
    consistent way to implement and use various sampling algorithms. Subclasses
    should implement the specific sampling logic.

    Methods:
    - __init__(self, *args, **kwargs): Initializes the sampling method with 
    any necessary parameters.
    - sample(self, *args, **kwargs): Generates samples according to the specific sampling method.
    - generate_samples(self): Returns the generated samples.
    """

    @property
    def n_s(self):
        """
        Number of samples getter
        """
        return self._ns

    @n_s.setter
    def n_s(self, value: int) -> int:
        """
        Number of samples setter
        """
        self._ns = value

    @property
    def var_limits(self):
        """
        Upper and lower bounds of variables getter
        """
        return self._var_limits

    @var_limits.setter
    def var_limits(self, value: np.ndarray) -> np.ndarray:
        """
        Upper and lower bounds of variables setter
        """
        self._var_limits = copy.deepcopy(value)

    @property
    def options(self):
        """
        Options dictionary getter
        """
        return self._options

    @options.setter
    def options(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Options dictionary setter
        """
        self._options = copy.deepcopy(value)

    def scale_to_limits(self, samples: np.ndarray) -> np.ndarray:
        """
          Scale the samples from the unit hypercube to the specified limit.
        """
        n = self.var_limits.shape[0]
        for i in range(n):
            samples[:, i] = self.var_limits[i, 0] + samples[:, i] * \
                (self.var_limits[i, 1] - self.var_limits[i, 0])
            if "msize" in self.options.keys():
                s = samples[:, i]
                nr = int((self.var_limits[i, 1] - self.var_limits[i, 0])/(self.options["msize"][i]
                         if isinstance(self.options["msize"], list) else self.options["msize"]))
                mod = s % ((self.var_limits[i, 1] - self.var_limits[i, 0])/nr)
                samples[:, i] = s - mod

        return samples

    @abstractmethod
    def generate_samples(self, ns: int):
        """ Compute the requested number of sampling points.
          The number of dimensions (nx) is determined based on `varLimits.shape[0].` """
        pass

    @abstractmethod
    def set_options(self):
        """
        Sampling method options setter
        """
        pass

    @abstractmethod
    def utilities(self):
        """
        Sampling method utilities function
        """
        pass

    @abstractmethod
    def methods(self):
        """
        Sampling method variants
        """
        pass

    @classmethod
    def normal_pdf(cls, x):
        """
        Normal pdf calculator
        """
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

    @classmethod
    def normal_cdf(cls, x):
        """
        Normal cdf calculator
        """
        # Approximation to CDF of standard normal distribution
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


class FullFactorial(Sampling):
    """
    Full factorial sampler.

    Generates a full factorial design over the provided variable limits.
    The sampler creates a grid of points that spans the entire design space,
    optionally weighting dimensions to control the number of levels per
    dimension.  It supports clipping to the exact number of requested samples
    and scales the unit‑hypercube points to the user‑specified bounds.

    Parameters
    ----------
    ns : int
        Desired number of samples (points) to generate.
    w : np.ndarray or None
        Optional weighting vector for each dimension.  If ``None`` all
        dimensions are weighted equally.
    c : bool
        If ``True`` the sampler will clip the generated grid to exactly
        ``ns`` points; otherwise the grid may contain more points than
        requested.
    vlim : np.ndarray
        Variable limits as a ``(n_dims, 2)`` array where each row contains
        ``[lower_bound, upper_bound]`` for a dimension.

    Notes
    -----
    The implementation determines the number of levels per dimension based
    on the weights and the total number of samples, then constructs the
    Cartesian product of evenly spaced points in each dimension.  The final
    design is scaled from the unit hypercube to the provided limits.
    """

    def __init__(self, ns: int, w: np.ndarray, c: bool, vlim: np.ndarray):
        self.options = {}
        self.options["weights"] = copy.deepcopy(w)
        self.options["clip"] = c
        self.var_limits = copy.deepcopy(vlim)
        self.n_s = ns

    def set_options(self, w: np.ndarray, c: bool, la: np.ndarray):
        self.options = {}
        self.options["weights"] = copy.deepcopy(w)
        self.options["clip"] = c
        self.options["limits"] = copy.deepcopy(la)

    def utilities(self):
        pass

    def methods(self):
        pass

    def generate_samples(self):
        npts = self.n_s
        nx = self.var_limits.shape[0]

        if self.options["weights"] is None:
            weights = np.ones(nx) / nx
        else:
            weights = np.atleast_1d(self.options["weights"])
            weights = np.divide(weights, np.sum(weights))

        num_list = np.ones(nx, int)
        while np.prod(num_list) < npts:
            ind = np.argmax(weights - num_list / np.sum(num_list))
            num_list[ind] += 1

        lins_list = [np.linspace(0.0, 1.0, num_list[kx]) for kx in range(nx)]
        x_list = np.meshgrid(*lins_list, indexing="ij")

        if self.options["clip"]:
            npts = np.prod(num_list)

        x = np.zeros((npts, nx))
        for kx in range(nx):
            x[:, kx] = x_list[kx].reshape(np.prod(num_list))[:npts]

        return self.scale_to_limits(x)


class LHS(Sampling):
    """
    Latin Hypercube Sampler.

    This sampler generates a Latin hypercube design, which provides a
    stratified sampling of the parameter space.  Each dimension is divided
    into *n* equally‑probable intervals and a single point is drawn from each
    interval, ensuring that the full range of each variable is explored while
    keeping the total number of samples relatively low.

    Parameters
    ----------
    sampling : SamplingTemplate
        An instance of a ``SamplingTemplate`` (or compatible object) that
        defines the dimensionality, bounds, and any additional constraints
        for the design space.

    Returns
    -------
    np.ndarray
        A 2‑D array of shape ``(n_samples, n_dimensions)`` containing the
        generated sample points.

    Notes
    -----
    - The implementation assumes that ``sampling`` provides ``n_samples`` and
      ``bounds`` attributes. Adjust the sampler if your template uses a
      different interface.
    - For reproducibility, set the random seed via ``numpy.random.seed`` before
      calling ``sample()``.
    """

    def __init__(self, ns: int, vlim: np.ndarray):
        """
        Generate samples within a specified parameters bounds.

        Parameters
        ----------
        ns : int
            Number of samples to generate.
        vlim : np.ndarray
            Array defining the lower and upper bounds for each parameters component.

        Returns
        -------
        LHS class instant.
        """
        self.options = {}
        self.options["criterion"] = "ExactSE"
        self.options["randomness"] = 10000
        self.n_s = ns
        self.var_limits = copy.deepcopy(vlim)

    def utilities(self):
        pass

    def set_options(self, c: str, r: Any):
        self.options["criterion"] = c
        self.options["randomness"] = r

    def generate_samples(self):
        nx = self.var_limits.shape[0]

        if isinstance(self.options["randomness"], np.random.RandomState):  # pylint: disable=no-member
            self.random_state = self.options["randomness"]
        elif isinstance(self.options["randomness"], int):
            # pylint: disable=no-member
            self.random_state = np.random.RandomState(
                self.options["randomness"])
        else:
            self.random_state = np.random.RandomState()  # pylint: disable=no-member

        if self.options["criterion"] != "ExactSE":
            return self.scale_to_limits(self.methods(
                nx,
                ns=self.n_s,
                criterion=self.options["criterion"],
                r=self.random_state,
            ))
        elif self.options["criterion"] == "ExactSE":
            return self.scale_to_limits(self.methods(nx, self.n_s))

    def methods(self, nx: int = None, ns: int = None, criterion: str = None, r: Any = None):
        if criterion is not None:
            if self.options["criterion"]:
                return lhs(
                    nx,
                    samples=ns,
                    criterion=self.options["criterion"],
                    iterations=10,
                    random_state=r
                )
            else:
                return lhs(
                    nx,
                    samples=ns,
                    criterion=self.options["criterion"],
                    random_state=r
                )
        else:
            return self._exactse(nx, ns)

    def _optimize_exact_se(self, xf, t0=None,
                           outer_loop=None, inner_loop=None, jj=20, tol=1e-3,
                           p=10, return_hist=False, fixed_index=[]):

        # Initialize parameters if not defined
        if t0 is None:
            t0 = 0.005 * self._phi_p(xf, p=p)
        if inner_loop is None:
            inner_loop = min(20 * xf.shape[1], 100)
        if outer_loop is None:
            outer_loop = min(int(1.5 * xf.shape[1]), 30)

        tt = t0
        xf_ = xf[:]  # copy of initial plan
        x_best = xf_[:]
        d = xf.shape[1]
        phip_ = self._phi_p(x_best, p=p)
        phip_best = phip_

        hist_t = list()
        hist_proba = list()
        hist_phip = list()
        hist_phip.append(phip_best)

        # Outer loop
        for z in range(outer_loop):
            phip_oldbest = phip_best
            n_acpt = 0
            n_imp = 0
            # Inner loop
            for i in range(inner_loop):
                modulo = (i + 1) % d
                l_x = list()
                l_phip = list()
                for j in range(jj):
                    l_x.append(xf_.copy())
                    l_phip.append(self._phi_p_transfer(l_x[j], k=modulo,
                                                       phi_p=phip_, p=p, fixed_index=fixed_index))
                l_phip = np.asarray(l_phip)
                k = np.argmin(l_phip)
                phip_try = l_phip[k]
                # Threshold of acceptance
                if phip_try - phip_ <= tt * self.random_state.rand(1)[0]:
                    phip_ = phip_try
                    n_acpt = n_acpt + 1
                    xf_ = l_x[k]
                    # Best plan retained
                    if phip_ < phip_best:
                        x_best = xf_
                        phip_best = phip_
                        n_imp = n_imp + 1
                hist_phip.append(phip_best)

            p_accpt = float(n_acpt) / inner_loop  # probability of acceptance
            p_imp = float(n_imp) / inner_loop  # probability of improvement

            hist_t.extend(inner_loop * [tt])
            hist_proba.extend(inner_loop * [p_accpt])

        if phip_best - phip_oldbest < tol:
            # flag_imp = 1
            if p_accpt >= 0.1 and p_imp < p_accpt:
                tt = 0.8 * tt
            elif p_accpt >= 0.1 and p_imp == p_accpt:
                pass
            else:
                tt = tt / 0.8
        else:
            # flag_imp = 0
            if p_accpt <= 0.1:
                tt = tt / 0.7
            else:
                tt = 0.9 * tt

        hist = {"PhiP": hist_phip, "T": hist_t, "proba": hist_proba}

        if return_hist:
            return x_best, hist
        else:
            return x_best

    def _phi_p(self, x, p=10):

        return ((pdist(x) ** (-p)).sum()) ** (1.0 / p)

    def _phi_p_transfer(self, x, k, phi_p, p, fixed_index):
        """ Optimize how we calculate the phi_p criterion. """

        # Choose two (different) random rows to perform the exchange
        i1 = self.random_state.randint(x.shape[0])
        while i1 in fixed_index:
            i1 = self.random_state.randint(x.shape[0])

        i2 = self.random_state.randint(x.shape[0])
        while i2 == i1 or i2 in fixed_index:
            i2 = self.random_state.randint(x.shape[0])

        x_ = np.delete(x, [i1, i2], axis=0)

        dist1 = cdist([x[i1, :]], x_)
        dist2 = cdist([x[i2, :]], x_)
        d1 = np.sqrt(
            dist1 ** 2 + (x[i2, k] - x_[:, k]) ** 2 - (x[i1, k] - x_[:, k]) ** 2
        )
        d2 = np.sqrt(
            dist2 ** 2 - (x[i2, k] - x_[:, k]) ** 2 + (x[i1, k] - x_[:, k]) ** 2
        )

        res = (phi_p ** p + (d1 ** (-p) - dist1 ** (-p) + d2 ** (-p) -
                             dist2 ** (-p)).sum()) ** (1.0 / p)
        x[i1, k], x[i2, k] = x[i2, k], x[i1, k]

        return res

    def _exactse(self, dim, nt, fixed_index=[], p0=[]):
        # Parameters of Optimize Exact Solution Evaluation procedure
        if len(fixed_index) == 0:
            p0 = lhs(dim, nt, criterion=None, random_state=self.random_state)
        else:
            p0 = p0
            self.random_state = np.random.RandomState()  # pylint: disable=no-member
        j = 20
        outer_loop = min(int(1.5 * dim), 30)
        inner_loop = min(20 * dim, 100)

        p, _ = self._optimize_exact_se(
            p0,
            outer_loop=outer_loop,
            inner_loop=inner_loop,
            jj=j,
            tol=1e-3,
            p=10,
            return_hist=True,
            fixed_index=fixed_index,
        )
        return p

    def expand_lhs(self, x, n_points, method="basic"):
        """
        Expand the LHC sampling
        """
        var_limits = self.options["varLimits"] if self.var_limits is None else self.var_limits

        new_num = len(x) + n_points

        # Evenly spaced intervals with the final dimension of the LHS
        intervals = []
        for i, _ in enumerate(var_limits):
            intervals.append(np.linspace(
                var_limits[i][0], var_limits[i][1], new_num + 1))

        # Creates a subspace with the rows and columns that have no points
        # in the new space
        subspace_limits = [[]] * len(var_limits)
        subspace_bool = []
        for i in range(len(var_limits)):
            subspace_limits[i] = []

            subspace_bool.append(
                [
                    [
                        intervals[i][j] < x[kk][i] < intervals[i][j + 1]
                        for kk in range(len(x))
                    ]
                    for j in range(len(intervals[i]) - 1)
                ]
            )

            [
                subspace_limits[i].append(
                    [intervals[i][ii], intervals[i][ii + 1]])
                for ii in range(len(subspace_bool[i]))
                if not (True in subspace_bool[i][ii])
            ]

        # Sampling of the new subspace
        sampling_new = LHS(ns=n_points, vlim=np.array([[0.0, 1.0]] * len(var_limits)))
        x_subspace = sampling_new.generate_samples()

        column_index = 0
        sorted_arr = x_subspace[x_subspace[:, column_index].argsort()]

        for j in range(len(var_limits)):
            for i in range(len(sorted_arr)):
                sorted_arr[i, j] = subspace_limits[j][i][0] + sorted_arr[i, j] * (
                    subspace_limits[j][i][1] - subspace_limits[j][i][0]
                )

        h_sorted = np.zeros_like(sorted_arr)
        for j in range(len(var_limits)):
            order = np.random.permutation(len(sorted_arr))
            h_sorted[:, j] = sorted_arr[order, j]

        x_new = np.concatenate((x, h_sorted), axis=0)

        if method == "ExactSE":
            # Sampling of the new subspace
            sampling_new = LHS(ns=n_points, vlim=var_limits)
            x_new = sampling_new._exactse(
                len(x_new), len(x_new), fixed_index=np.arange(0, len(x), 1), p0=x_new
            )

        return x_new


class RS(Sampling):
    """
    Random sampling

    :param Sampling: uses sampling template
    :type Sampling: _type_
    """

    def __init__(self, ns: int, vlim: np.ndarray, options: Dict[str, Any] = {}):
        self.options = options
        self.n_s = ns
        self.var_limits = copy.deepcopy(vlim)

    def generate_samples(self):
        nx = self.var_limits.shape[0]
        if self.options != {} and "randomness" in self.options:
            np.random.seed(self.options["randomness"])
        return self.scale_to_limits(np.random.rand(self.n_s, nx))

    def methods(self):
        pass

    def utilities(self):
        pass

    def set_options(self):
        pass


class Halton(Sampling):
    """
    Hammerseley or Halton sequence sampling

    :param Sampling: uses sampling template
    :type Sampling: _type_
    """

    def __init__(self, ns: int, vlim: np.ndarray, is_ham: bool = True):
        """
        Initializer

        :param ns: number of samples
        :type ns: int
        :param vlim: Variables upper and lower bounds
        :type vlim: np.ndarray
        :param is_ham: is hamming distance, defaults to True
        :type is_ham: bool, optional
        """
        self.options = {}
        self.n_s = ns
        self.var_limits = copy.deepcopy(vlim)
        self.ishammersley = is_ham

    def prime_generator(self, n: int):
        """
        Docstring for prime_generator

        :param self: Description
        :param n: number of primes to be generated
        :type n: int
        """
        prime_list = []
        current_no = 2
        if n < 0:
            raise ValueError("A negative count is provided which might lead to an infinite loop. Assign the count to a positive value.")
        while len(prime_list) < n:
            for i in range(2, current_no):
                if (current_no % i) == 0:
                    break
            else:
                prime_list.append(current_no)
            current_no += 1
        return prime_list

    def base_conv(self, a, b):
        """
        Base converter
        """
        string_representation = []
        if a < b:
            string_representation.append(str(a))
        else:
            while a > 0:
                a, c = (a // b, a % b)
                string_representation.append(str(c))
            string_representation = string_representation[::-1]
        return string_representation

    def data_sequencing(self, pb):
        """
        Data sequencing
        """
        pure_numbers = np.arange(0, self.n_s)
        bitwise_rep = []
        reversed_bitwise_rep = []
        sequence_bitwise = []
        sequence_decimal = np.zeros((self.n_s, 1))
        for i in range(0, self.n_s):
            base_rep = self.base_conv(pure_numbers[i], pb)
            bitwise_rep.append(base_rep)
            reversed_bitwise_rep.append(base_rep[::-1])
            sequence_bitwise.append(['0.'] + reversed_bitwise_rep[i])
            sequence_decimal[i, 0] = self.pb_to_dec(sequence_bitwise[i], pb)
        sequence_decimal = sequence_decimal.reshape(sequence_decimal.shape[0], )
        return sequence_decimal

    def pb_to_dec(self, num, base):
        """
        Convert to decimal
        """
        binary = num
        decimal_equivalent = 0
        # Convert fractional part decimal equivalent
        for i in range(1, len(binary)):
            decimal_equivalent += int(binary[i]) / (base ** i)
        return decimal_equivalent

    def primes_from_2_to(self, n):
        """Prime number from 2 to n.
        From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
        :param int n: sup bound with ``n >= 6``.
        :return: primes in 2 <= p < n.
        :rtype: list
        """
        sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
        for i in range(1, int(n ** 0.5) // 3 + 1):
            if sieve[i]:
                k = 3 * i + 1 | 1
                sieve[k * k // 3::2 * k] = False
                sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
        return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

    def van_der_corput(self, n_sample, base=2):
        """Van der Corput sequence.
        :param int n_sample: number of element of the sequence.
        :param int base: base of the sequence.
        :return: sequence of Van der Corput.
        :rtype: list (n_samples,)
        """
        sequence = []
        for i in range(n_sample):
            n_th_number, denom = 0., 1.
            while i > 0:
                i, remainder = divmod(i, base)
                denom *= base
                n_th_number += remainder / denom
            sequence.append(n_th_number)

        return sequence

    def generate_samples(self, RS=None):
        """Halton sequence.
        :param int dim: dimension
        :param int n_sample: number of samples.
        :return: sequence of Halton.
        :rtype: array_like (n_samples, n_features)
        """
        if self.ishammersley:
            no_features = self.var_limits.shape[0]
            # Generate list of no_features prime numbers
            prime_list = self.prime_generator(no_features)
            sample = np.zeros((self.n_s, no_features))
            for i in range(0, no_features):
                sample[:, i] = self.data_sequencing(prime_list[i])
            # Scale input data, then find data points closest in sample space.
            # Unscale before returning points
            min_ = np.min(self.var_limits, axis=1)
            max_ = np.max(self.var_limits, axis=1)
            sample = sample * (max_ - min_) + min_
        else:
            big_number = 10
            dim = self.var_limits.shape[0]
            while 'Not enought primes':
                base = self.primes_from_2_to(big_number)[:dim]
                if len(base) == dim:
                    break
                big_number += 1000

            # Generate a sample using a Van der Corput sequence per dimension.
            sample = [self.van_der_corput(self.n_s + 1, dim) for dim in base]
            sample = np.stack(sample, axis=-1)[1:]
            min_ = np.min(self.var_limits, axis=1)
            max_ = np.max(self.var_limits, axis=1)
            sample = sample * (max_ - min_) + min_

        return sample

    def methods(self):
        pass

    def utilities(self):
        pass

    def set_options(self):
        pass


@dataclass
class KernelFunctions:
    """
    A data class that holds a dictionary for all the kernels implemented in this package 
    mapped to their corresponding callables
    """
    kfs: Dict[str, KERNEL] = None

    def __init__(self):
        """
        Initializer
        """
        self.kfs = {
            "Linear": Linear,
            "Gaussian": Gaussian,
            "Epanechnikov": Epanechnikov,
            "Cosine": Cosine,
            "UniformRectangular": UniformRectangular,
            "Triweight": Triweight,
            "Tricube": Tricube,
            "Silverman": Silverman,
            "Sigmoid": Sigmoid,
            "Biweight": Biweight,
            "Logistic": Logistic,
            "Laplace": Laplace,
            "GaussianRBF": GaussianRBF,
            "MultiquadricRBF": MultiquadricRBF,
            "InverseMultiquadricRBF": InverseMultiquadricRBF,
            "ThinPlateSplineRBF": ThinPlateSplineRBF,
            "Cauchy": Cauchy
        }


class ActiveSampling(Sampling):
    """
    Active sampling based on KDE and incumbents distribution

    :param Sampling: uses sampling template
    :type Sampling: _type_
    """

    def __init__(self, data: np.ndarray, n_r: int, vlim: np.ndarray,
                 kernel_type: List[str] = ["Gaussian"], bw_method=TUNING_METHOD.SCOTT.name,
                 seed: int = 10000, weights: Any = None, h: List[float] = None):
        """
        Initializer

        :param data: data points (usually incumbents, critical incumbents, or nondominated points)
        :type data: np.ndarray
        :param n_r: number of reduced dimensions
        :type n_r: int
        :param vlim: variable bounds
        :type vlim: np.ndarray
        :param bw_method: bandwidth rule of thumb, defaults to TUNING_METHOD.SCOTT.name
        :type bw_method: _type_, optional
        :param weights: datapoints weight, defaults to None
        :type weights: Any, optional
        :param h: initial bandwidth, defaults to None
        :type h: List[float], optional
        """
        self._cov_decomp: str = 'svd'
        self.reducer: Reducers = None
        self.data_reduced: np.ndarray = None
        self.data_standardized: np.ndarray = None
        self.data: np.ndarray = np.atleast_2d(np.asarray(data))
        self._msgs: List[List[str]] = []
        self.kernel_funcs: Dict = KernelFunctions().kfs
        if self.data.shape[0] <= 1:
            self._msgs.append([2, "`data` passed in to the \
                               `activeSampling` constructor should \
                               include multiple sample points."])
            raise ValueError("`data` passed in to the `activeSampling` \
                             constructor should include multiple sample points.")
        self.n_r: int = n_r
        self.n_s, self.n_d = self.data.shape
        if weights is not None:
            self._weights: Any = np.atleast_1d(weights).astype(float)
            self._weights /= sum(self._weights)
            if self._weights.ndim != 1:
                self._msgs.append([2, "`weights` passed in to the `activeSampling` \
                                   constructor should be on-diemsional vector."])
                raise ValueError("`weights` passed in to the `activeSampling` \
                                   constructor should be on-diemsional vector.")
            if self._weights.shape[0] != self.n_s:
                self._msgs.append([2, "`weights` passed in to the `activeSampling` \
                                   constructor should have the same size of \
                                   the input `data` array."])
                raise ValueError("`weights` passed in to the `activeSampling` \
                                 constructor should have the same size of the input `data` array.")
            self._ne = 1/sum(self._weights**2)
        else:
            self._weights: Any = np.atleast_1d([1]*self.n_s).astype(float)
            self._weights /= sum(self._weights)
            self._ne = int(1/sum(self._weights**2))

        self.resampled_data: np.ndarray = np.zeros((self.n_r, self.n_d))
        self.seed = seed
        self.var_limits = copy.deepcopy(vlim)
        self.data_normalized = self.data / (vlim[:, 1] - vlim[:, 0])
        if self.n_d > 3:
            self.reducer = Reducers(data=data, vlim=self.var_limits, nd=self.n_d)
            self.reducer.rd()
            data = self.reducer.data_reduced
        else:
            data = self.data
        self.kernel: List[KERNEL] = []
        for k in kernel_type:
            if k in self.kernel_funcs.keys():
                kf = self.kernel_funcs.get(k)
                self.kernel.append(kf(data=data, vlim=vlim, weights=self._weights,
                                      bw_method=bw_method, h=h[:data.shape[1]], n_r=self.n_r))
            else:
                self._msgs.append([1, "Unknown kernel type. \
                                   Switched to the default Gaussian kernel."])
                self.kernel.append(Gaussian(data=data, vlim=vlim, weights=self._weights,
                                            bw_method=bw_method, h=h[:data.shape[1]], n_r=self.n_r))

    def standardize_data(self):
        """
        Standardize data
        """
        self.data_scaled = (self.data - self.var_limits[:, 0]) / \
            (self.var_limits[:, 1] - self.var_limits[:, 0])
        self.means = np.mean(self.data_scaled, axis=0)
        self.std_devs = np.std(self.data_scaled, axis=0)
        self.data_standardized = (self.data_scaled - self.means) / self.std_devs
        nan_indices = np.isnan(self.data_standardized)
        col_means = np.random.normal(0, 1e-5, size=self.n_d)
        self.data_standardized[nan_indices] = np.take(col_means, np.where(nan_indices)[1])

    def rd(self):
        """
        Reduce the space dimensionality to principal components
        """
        # Compute the covariance matrix
        self.standardize_data()
        cov_matrix = np.cov(self.data_standardized, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues in descending order, and rearrange the eigenvectors accordingly
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues_sorted = eigenvalues[sorted_indices]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]

        # Select the top 'k' eigenvectors to form the new matrix
        self.k = 3  # Number of principal components we want (reduce to 3D)
        self.eigenvectors_top_k = eigenvectors_sorted[:, :self.k]

        # Project the original data onto the new space
        self.data_reduced = np.dot(self.data_standardized, self.eigenvectors_top_k)

    def project_rd_to_original_space(self, samples: np.ndarray):
        """
        Project reduced space to original full space
        """
        return samples.dot((self.eigenvectors_top_k[:, :self.k].T) + self.means)

    @classmethod
    def kde_resample(cls, ks: List[KERNEL],
                     weights: List[float], data: List[List[float]],
                     seed: int):
        """
        Resample using inferred distributions by introduced KDEs for a given dataset 
        """
        random.seed(seed)
        base_point = random.choice(data)
        # Select kernel based on weighted probability
        kernel = random.choices(ks, weights=weights)[0]
        noise = kernel._sample_noise()
        return [b + n for b, n in zip(base_point, noise)]

    def resample(self, size=None):
        """
        Resample based on inferred distributions
        """
        samples = []
        if size is None:
            size = self.n_r
        for _ in range(size):
            samples += self.kde_resample(ks=self.kernel,
                                         weights=[1/len(self.kernel)]*len(self.kernel),
                                         data=self.data_reduced if self.n_d > 3
                                         else self.data, seed=self.seed)
        # for i, _ in enumerate(self.kernel):
        #     if size is None:
        #         size = int(self.kernel[i]._ne)

        #     # if size > self.kernel[i]._points.shape[0]:
        #     #     size = self.kernel[i]._points.shape[0]

        #     if self.kernel[i].est_pdf is None:
        #         self.kernel[i].est_pdf = self.kernel[i].estimate_pdf()
        #     if np.any(np.isnan(self.kernel[i].est_pdf)):
        #         for j, _ in enumerate(self.kernel[i].est_pdf):
        #             if np.isnan(self.kernel[i].est_pdf[j]):
        #                 self.kernel[i].est_pdf[j] = 0
        #         if sum(self.kernel[i].est_pdf) <= 0:
        #             self.kernel[i].est_pdf = \
        #                 np.atleast_1d([1/len(self.kernel[i].est_pdf)]*\
        #                               len(self.kernel[i].est_pdf))
        #         else:
        #             self.kernel[i].est_pdf /= sum(self.kernel[i].est_pdf)

        # densities = [k.est_pdf for k in self.kernel]

        # # Average the densities for sampling
        # combined_density = np.mean(densities, axis=0)

        # # Normalize the combined density
        # combined_density /= np.sum(combined_density)

        # # Sample from the combined density
        # sampled_indices = np.random.choice(self.kernel[0]._points.shape[0], \
        #                                    size=size, p=combined_density)
        if self.n_d <= 3:
            return np.array(samples)
        else:
            x_resampled = \
                self.reducer.project_rd_to_original_space(samples=np.array(samples))
            x_resampled_clipped = np.clip(x_resampled, 0, 1)
            x_resampled_original = x_resampled_clipped * (self.var_limits[:, 1] -
                                                          self.var_limits[:, 0]) + \
                self.var_limits[:, 0]
            return x_resampled_original

        # rg = np.random
        # random_state = np.random.Generator(rg.PCG64DXSM(seed=seed))
        # if type(self.kernel[i]).__name__ == "Gaussian" and self.kernel[i]._cov is not None:
        #   MVN = CustomMultivariateNormal(np.zeros((self.n_d,), np.float64),
        # self.kernel[i]._cov, seed)
        #   normDist = MVN.sample(size)
        # The following numpy bug shows a high risk that multivariate_normal gives
        # different results when numpy linear algebra solvers
        # get updated (mainly matrix factorization and decomposition methods) and/or
        # when the bitgenerators that control sources
        # of randomization have new update which is less likely to happen than the former.
        # For those reasons, it is not recommended to fully rely on numpy statistical
        # library to sample points via multivariate normal
        # distribution given a covariance matrix and mean values.
        # So customed methods are developed here starting from release no. 2408
        # https://github.com/numpy/numpy/issues/22975
        # random_state = np.random.RandomState(seed)
        # if type(self.kernel[i]).__name__ == "Gaussian" and self.kernel[i]._cov is not None:
        #   normDist = np.transpose(random_state.multivariate_normal(
        #       np.zeros((self.n_d,), float), self.kernel[i]._cov, size=size
        #   ))
        #   indices = random_state.choice(self.kernel._points.shape[0],
        # size=size, p=self.kernel.est_pdf)
        #   means = self.data[indices, :]
        #   if isinstance(normDist, stats._multivariate.multivariate_normal_frozen):
        #     new = means + normDist.rvs(size=size, random_state=random_state)
        # #   ))
        #   else:
        #     new = means + normDist
        # else:
        #   indices1 = random_state.choice(self.kernel.data.shape[0], size=size)
        #   indices2 = random_state.choice(self.kernel._points.shape[0],
        # size=size, p=abs(self.kernel.est_pdf))
        #   means1 = self.data[indices1, :]
        #   means2 = self.kernel._points[indices2, :]
        #   new = (means1 + means2)/2

        # omit = []
        # for i in range(size):
        #   for j in range(self.n_d):
        #     if new[i,j] < self.varLimits[j, 0] or new[i,j] > self.varLimits[j, 1]:
        #       omit.append(i)

        # return np.delete(new, omit, axis=0)
    def generate_samples(self, ns: int):
        pass

    def set_options(self):
        pass

    def utilities(self):
        pass

    def methods(self):
        pass


class TunablePSS(Sampling):
    """
    Tunable particle swar sampler

    :param Sampling: uses sampling template
    :type Sampling: _type_
    """

    def __init__(self, data: np.ndarray, y: np.ndarray, x_inc: np.ndarray,
                 it: int, vlim: np.ndarray, num_particles=100, max_iter=100,
                 inertia_weight: int = 0.1, cognitive_weight: int = 1,
                 social_weight: int = 1, seed: int = 10000, weights: Any = None):
        self._cov_decomp: str = 'svd'
        self._msgs: List[List[str]] = []
        self.it = it
        self.data: np.ndarray = copy.deepcopy(data)
        self.x_incumbent: np.ndarray = copy.deepcopy(x_inc)
        if self.data.size <= 1:
            self._msgs.append([2, "`data` passed in to the `activeSampling` \
                               constructor should include multiple sample points."])
            raise ValueError("`data` passed in to the `activeSampling` \
                             constructor should include multiple sample points.")
        self.selector: List[RF] = [RF(n_estimators=10, max_depth=3)] * y.shape[1]
        self.n_s, self.n_d = self.data.shape
        if weights is None:
            for i in range(y.shape[1]):
                self.selector[i].fit(data, y[:, i])
                if i == 0:
                    weights = self.selector[i].get_feature_importance(data, y[:, i])
                else:
                    weights += self.selector[i].get_feature_importance(data, y[:, i])

        self._weights = np.atleast_1d(weights).astype(float)
        self._weights /= sum(self._weights) if sum(self._weights) > 0 else 1
        if not self._weights.ndim == 1:
            self._msgs.append([2, "`weights` passed in to the `activeSampling` "
                               "constructor should be on-diemsional vector."])
            raise ValueError("`weights` passed in to the `activeSampling` \
                                constructor should be on-diemsional vector.")
        if self._weights.shape[0] != self.n_d:
            self._msgs.append([2, "`weights` passed in to the `activeSampling` \
                                constructor should have the same size of \
                                the input `data` array."])
            raise ValueError("`weights` passed in to the `activeSampling` \
                                constructor should have the same size of the input `data` array.")
        self._ne = 1/sum(self._weights**2)

        self.seed = seed
        self.var_limits = [[], []]
        for v in vlim:
            self.var_limits[0].append(v[0])
            self.var_limits[1].append(v[1])
        self.var_limits = np.array(self.var_limits)
        self.data_normalized = self.data / (vlim[:, 1] - vlim[:, 0])
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia_weight = inertia_weight
        self.social_weight = social_weight
        self.cognitive_weight = cognitive_weight

    def generate_samples(self, ns: int):
        pass

    def set_options(self):
        pass

    def utilities(self):
        pass

    def methods(self):
        pass

    def target_distribution(self, x):
        """Multivariate Gaussian distribution."""
        mean = np.zeros(x.shape[0])  # Mean at the origin
        cov = np.eye(x.shape[0])  # Identity covariance matrix (standard normal)
        exponent = -0.5 * np.dot((x - mean), np.dot(np.linalg.inv(cov), (x - mean)))
        return (1 / np.sqrt((2 * np.pi)**len(x) * np.linalg.det(cov))) * np.exp(exponent)

    def particle_swarm_sampling(self, size):
        """
        PSS sampling
        """
        particles = [Particle(bounds=self.var_limits, pos=self.data[i]
                              if i < len(self.data) else None) for i in range(self.num_particles)]
        global_best_position = None
        global_best_value = float('-inf')

        samples = []

        for _ in range(self.max_iter):
            for particle in particles:
                particle.evaluate(self.target_distribution)
                if particle.best_value > global_best_value:
                    global_best_value = particle.best_value
                    global_best_position = particle.best_position

            for particle in particles:
                particle.update_velocity(global_best_position, self.inertia_weight,
                                         self.cognitive_weight, self.social_weight)
                particle.update_position(self.var_limits)

            # Store the current positions as samples
            samples.extend([particle.position for particle in particles])

        for i, _ in enumerate(samples):
            for j, _ in enumerate(samples[i]):
                if samples[i][j] < self.var_limits[0, j]:
                    samples[i][j] = self.var_limits[0, j]
                if samples[i][j] > self.var_limits[1, j]:
                    samples[i][j] = self.var_limits[1, j]

        return np.array(samples[:size])

    def resample(self, size=None, seed=None):
        """
        Resample based on the inferred distribution of input data points
        """
        # Run Multivariate Particle Swarm Sampling
        self.seed = copy.deepcopy(seed)
        samples = self.particle_swarm_sampling(size)

        res = self.resample_multidimensional_variables(variables=samples,
                                                       dimension_weights=self._weights,
                                                       num_samples=size) \
            if sum(self._weights) > 0 \
            else samples
        # self.plotting(res)
        return res

    def plotting(self, data):
        """
        Plotting data points
        """
        # Generate random data for 10-dimensional parameters (100 samples)
        num_dimensions = len(data[0])

        # Convert the data into a DataFrame for better handling with seaborn
        df = pd.DataFrame(data, columns=[f"x{i+1}" for i in range(num_dimensions)])

        # Create a pair plot (scatter plot matrix)
        sns.pairplot(df, height=2.5)

        # Show the plot
        plt.suptitle(f"Pairwise Scatter Plot Matrix of {num_dimensions} Dimensions", y=1.02)
        plt.show()

    def resample_multidimensional_variables(self, variables, dimension_weights, num_samples):
        """
        Randomly resamples multidimensional variables based on the \
            importance weight of each dimension.

        Parameters:
        - variables: A 2D numpy array (N x D), where N is \
            the number of variables and D is the number of dimensions.
        - dimension_weights: A 1D numpy array of shape (D,), \
            representing the importance of each dimension.
        - num_samples: The number of variables to sample.

        Returns:
        - A numpy array of shape (num_samples, D) containing the resampled variables.
        """
        # Normalize the importance weights for each dimension
        normalized_weights = dimension_weights / np.sum(dimension_weights)

        # Get the number of variables (N) and number of dimensions (D)
        num_variables, num_dimensions = variables.shape

        # Initialize an array to hold the resampled variables
        resampled_variables = np.tile(self.x_incumbent, (num_samples, 1))

        # For each dimension, we will sample based on the normalized importance weights
        for d in range(num_dimensions):
            # Calculate the cumulative distribution for the current dimension's weights
            cumulative_weights = np.cumsum(normalized_weights)

            # For each sample, generate a random number and map it to the corresponding index
            for i in range(num_samples):
                # Generate a random number between 0 and 1
                rd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(self.it+i)))  # pylint: disable=no-member
                rand_value = rd.rand()

                # Find the index where this random value fits within the cumulative distribution
                # This gives the index for the variable from which \
                # we'll sample the value for this dimension
                sample_index = np.searchsorted(cumulative_weights, rand_value)

                # Assign the corresponding value for this dimension
                resampled_variables[i, sample_index] = variables[i, d]

        return np.unique(resampled_variables, axis=0)


class TunableSA(Sampling):
    """
    Tunable Simulated Annealing

    :param Sampling: uses sampling template
    :type Sampling: _type_
    """

    def __init__(self, data: np.ndarray, y: np.ndarray,
                 x_inc: np.ndarray, it: int,
                 vlim: np.ndarray, initial_temp=100,
                 cooling_rate=0.99, max_iter=100,
                 seed: int = 10000, weights: Any = None):
        """
        Initializer
        """
        self.n_r: int = 0
        self._msgs: List[List[str]] = []
        rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(it+1)))  # pylint: disable=no-member
        self.it = it
        self.data: np.ndarray = copy.deepcopy(data)
        self.x_incumbent = copy.deepcopy(x_inc)

        self.selector = [RF(n_estimators=10, max_depth=3)] * y.shape[1]

        self.n_s, self.n_d = data.shape
        if weights is None:
            for i in range(y.shape[1]):
                self.selector[i].fit(data, y[:, i])
                if i == 0:
                    weights = self.selector[i].get_feature_importance(data, y[:, i])
                else:
                    weights += self.selector[i].get_feature_importance(data, y[:, i])

        self._weights = np.atleast_1d(weights).astype(float)
        self._weights /= sum(self._weights) if sum(self._weights) > 0 else 1
        if self._weights.ndim != 1:
            self._msgs.append([2, "`weights` passed in to the `activeSampling` \
                                constructor should be on-diemsional vector."])
            raise ValueError("`weights` passed in to the `activeSampling` \
                                constructor should be on-diemsional vector.")
        if self._weights.shape[0] != self.n_d:
            self._msgs.append([2, "`weights` passed in to the `activeSampling` \
                                constructor should have the same size of \
                                the input `data` array."])
            raise ValueError("`weights` passed in to the `activeSampling` \
                                constructor should have the same size of the input `data` array.")
        self._ne = 1/sum(self._weights**2)

        self.seed = seed
        self.var_limits = [[], []]
        for v in vlim:
            self.var_limits[0].append(v[0])
            self.var_limits[1].append(v[1])
        self.var_limits = np.array(self.var_limits)
        self.data_normalized = self.data / (vlim[:, 1] - vlim[:, 0])
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter

    def generate_samples(self, ns: int):
        pass

    def set_options(self):
        pass

    def utilities(self):
        pass

    def methods(self):
        pass

    def target_distribution(self, x):
        """Multivariate Gaussian distribution."""
        mean = np.zeros(x.shape[0])  # Mean at the origin
        cov = np.eye(x.shape[0])     # Identity covariance matrix (standard normal)
        return (1 / np.sqrt((2 * np.pi)**len(x) * np.linalg.det(cov))) * \
            np.exp(-0.5 * np.dot((x - mean), np.dot(np.linalg.inv(cov), (x - mean))))

    def get_neighbor(self, current_point, step_size=0.1):
        """
        Get a neighbor candidate
        """
        # Perturb each dimension by a small random amount
        return [x + np.random.uniform(-step_size, step_size) for x in current_point]

    def acceptance_probability(self, temperature):
        """Accept with a probability that decreases as temperature lowers."""
        return np.random.random() < \
            math.exp(-1 / temperature)  # Example: random acceptance, depending on temperature

    def simulated_annealing_sampling(self, size):
        """
        SA sampler run
        """
        current_point = self.x_incumbent
        samples = [current_point]

        # Start with an initial temperature
        temperature = self.initial_temp

        # Sampling loop: perform for num_samples iterations
        for _ in range(self.n_s):
            # Generate a neighboring sample by perturbing each dimension
            neighbor_point = self.get_neighbor(current_point,
                                               min(self.var_limits[1]-self.var_limits[0])/10)

            # Accept or reject the new point based on the acceptance criterion
            if self.acceptance_probability(temperature):
                current_point = neighbor_point

            # Store the accepted sample
            samples.append(current_point)

            # Cool down the temperature
            temperature *= self.cooling_rate

        spoints = [s*(self.var_limits[1, :] - self.var_limits[0, :]) -
                   self.var_limits[0, :] for s in samples]
        for i, _ in enumerate(spoints):
            for j, _ in enumerate(spoints[i]):
                if spoints[i][j] < self.var_limits[0, j]:
                    spoints[i][j] = self.var_limits[0, j]
                if spoints[i][j] > self.var_limits[1, j]:
                    spoints[i][j] = self.var_limits[1, j]
        return np.array(spoints[:size] if len(spoints) > size else spoints)

    def resample(self, size=None, seed=None):
        """
        Resample using simulated_annealing_sampling
        """
        self.seed = copy.deepcopy(seed)
        samples = self.simulated_annealing_sampling(size)

        res = self.resample_multidimensional_variables(variables=samples,
                                                       dimension_weights=self._weights,
                                                       num_samples=size) \
            if sum(self._weights) > 0 \
            else samples
        # self.plotting(res)
        return res

    def plotting(self, data):
        """ Plotting data points"""

        # Generate random data for 10-dimensional parameters (100 samples)
        num_dimensions = len(data[0])

        # Convert the data into a DataFrame for better handling with seaborn
        df = pd.DataFrame(data, columns=[f"x{i+1}" for i in range(num_dimensions)])

        # Create a pair plot (scatter plot matrix)
        sns.pairplot(df, height=2.5)

        # Show the plot
        plt.suptitle(f"Pairwise Scatter Plot Matrix of {num_dimensions} Dimensions", y=1.02)
        plt.show()

    def resample_multidimensional_variables(self, variables, dimension_weights, num_samples):
        """
        Randomly resamples multidimensional variables based on 
        the importance weight of each dimension.

        Parameters:
        - variables: A 2D numpy array (N x D), where N is 
        the number of variables and D is the number of dimensions.
        - dimension_weights: A 1D numpy array of shape (D,), 
        representing the importance of each dimension.
        - num_samples: The number of variables to sample.

        Returns:
        - A numpy array of shape (num_samples, D) containing the resampled variables.
        """
        # Normalize the importance weights for each dimension
        normalized_weights = dimension_weights / np.sum(dimension_weights)

        # Get the number of variables (N) and number of dimensions (D)
        num_variables, num_dimensions = variables.shape

        # Initialize an array to hold the resampled variables
        resampled_variables = np.tile(self.x_incumbent, (num_samples, 1))

        # For each dimension, we will sample based on the normalized importance weights
        for d in range(num_dimensions):
            # Calculate the cumulative distribution for the current dimension's weights
            cumulative_weights = np.cumsum(normalized_weights)

            # For each sample, generate a random number and map it to the corresponding index
            for i in range(len(variables)):
                # Generate a random number between 0 and 1
                # pylint: disable=no-member
                rd = np.random.RandomState(np.random.MT19937(
                    np.random.SeedSequence(self.it+i)))
                rand_value = rd.rand()

                # Find the index where this random value fits within the cumulative distribution
                # This gives the index for the variable from which
                # we'll sample the value for this dimension
                sample_index = np.searchsorted(cumulative_weights, rand_value)

                # Assign the corresponding value for this dimension
                resampled_variables[i, sample_index] = variables[i, d]

        return np.unique(resampled_variables, axis=0)


class BayesianActiveSampling(Sampling):
    """
    Active sampling based on Bayesian framework that utilizes expected improvement approach

    :param Sampling: uses sampling template
    :type Sampling: _type_
    """

    def __init__(self, data: np.ndarray, f_values: np.ndarray,
                 n_r: int, vlim: np.ndarray, kernel_type: dict = {"Gaussian": 1},
                 bw_method=TUNING_METHOD.SCOTT.name, seed: int = 10000,
                 weights: Any = None, h: List[float] = None):
        self._data_training: np.ndarray = None
        self._data_testing: np.ndarray = None
        self._data_validating: np.ndarray = None
        self._f_training: np.ndarray = None
        self._f_testing: np.ndarray = None
        self._f_validating: np.ndarray = None
        self._cov_decomp: str = 'svd'
        self.reducer: Reducers = None
        self.data_reduced: np.ndarray = None
        self.data_standardized: np.ndarray = None

        self.data: np.ndarray = np.atleast_2d(np.asarray(data))
        self.f_values = np.atleast_1d(np.asarray(f_values))
        self.kernel_funcs: Dict = KernelFunctions().kfs
        self._msgs: List[List[str]] = []
        self.n_s, self.n_d = self.data.shape
        if self.data is None or self.data.size <= 5:
            self._msgs.append([2, "`data` passed in to the `activeSampling` "
                               "constructor should include multiple sample points, "
                               "at least five sample points."])
            raise ValueError("`data` passed in to the `activeSampling` "
                             "constructor should include multiple sample points, "
                             "at least five sample points.")

        self.n_r = n_r

        if weights is not None:
            self._weights = np.atleast_1d(weights).astype(float)
            self._weights /= sum(self._weights)
            if not self._weights.ndim == 1:
                self._msgs.append([2, "`weights` passed in to the `activeSampling` "
                                   "constructor should be on-diemsional vector."])
            if not len(self._weights) == self.n_s:
                self._msgs.append([2, "`weights` passed in to the `activeSampling` "
                                   "constructor should have the same size of the input `data` array."])
                raise ValueError("`weights` passed in to the `activeSampling` "
                                 "constructor should have the same size of the input `data` array.")
            self._ne = 1/sum(self._weights**2)
        else:
            self._weights = None

        self.resampled_data = np.zeros((self.n_r, self.n_d))
        self.seed = seed
        self.var_limits = copy.deepcopy(vlim)
        self.split_dataset(xf=self.data[0:-1], y=self.f_values[0:-1], random_seed=seed)
        self.data_normalized = self.data / (vlim[:, 1] - vlim[:, 0])
        data_training = self._data_training
        self.cov = np.cov(self._data_training, rowvar=False)  # Shape (d, d)
        self._cov: np.ndarray = None
        self.bandwidth = h[0]
        # scaled_cov = self.bandwidth ** 2 * self.cov
        # self.inv_cov = np.linalg.inv(scaled_cov)

        try:
            self.inv_cov = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            self.inv_cov = np.linalg.pinv(self.cov)  # fallback to pseudo-inverse

        self.kernel: List[KERNEL] = []
        self.kernels: Dict[str, float] = kernel_type
        for k in kernel_type.keys():
            if k in self.kernel_funcs.keys():
                kf = self.kernel_funcs.get(k)
                self.kernel.append(kf(data=data_training, weights=self._weights,
                                      bw_method=bw_method, h=h))
            else:
                self._msgs.append([1, "Unknown kernel type. "
                                   "Switched to the default Gaussian kernel."])
                self.kernel.append(Gaussian(data=data_training,
                                            weights=self._weights, bw_method=bw_method, h=h))

    def split_dataset(self, xf, y, train_ratio=0.9, val_ratio=0.0,
                      test_ratio=0.1, shuffle=True, random_seed=None):
        """
        Split the data set into training, testing and validation data sets
        """
        assert len(xf) == len(y), "X and y must have the same number of samples."
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1."

        n_samples = xf.shape[0]

        # Shuffle indices
        if shuffle:
            if random_seed is not None:
                np.random.seed(random_seed)
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        # Compute split indices
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        # Apply splits to X and y
        self._data_training, self._f_training = xf[train_idx], y[train_idx]
        self._data_training = np.vstack((self._data_training, self.data[-1]))
        self._f_training = np.concatenate((self._f_training, np.array([self.f_values[-1]])))
        self._data_validating, self._f_validating = xf[val_idx], y[val_idx]
        self._data_testing, self._f_testing = xf[test_idx], y[test_idx]

    # Combined kernel using weighted average
    def _combined_kernel(self, x, xi):
        diff = xi - x
        # Mahalanobis distance squared
        r2 = diff.T
        total = 0.0
        weight_sum = sum(self.kernels.values())
        kernel_func_w = None
        for i, k in enumerate(self.kernel):
            kernel_func_w = self.kernels.get(k.__class__.__name__)
            k.set_bw("SILVERMAN")
            if kernel_func_w is not None:
                total += kernel_func_w * k.kf_multivar(r2)
        return total / weight_sum if weight_sum > 0 else 0.0

    def estimate_local_f_and_uncertainty(self, x, f_values, data):
        """
        Estimate the function value locally and its uncertainty
        """
        x = np.asarray(x)
        weights = []

        for xi in data:
            xi = np.asarray(xi)
            # Apply kernel on Mahalanobis distance squared
            w = self._combined_kernel(x, xi)
            weights.append(w)

        total_weight = sum(weights)
        if total_weight == 0:
            return float('inf'), float('inf')

        mean = sum(w * f for w, f in zip(weights, f_values)) / total_weight
        var = sum(w * (f - mean) ** 2 for w, f in zip(weights, f_values)) / total_weight
        return mean, var

    # def estimate_local_f_and_uncertainty(self, x, f_values):
    #     weights = []
    #     for xi in self.data:
    #         sqdist = sum((xi[j] - x[j]) ** 2 for j in range(self.d)) / (self.bandwidth ** 2)
    #         w = self._combined_kernel(sqdist)
    #         weights.append(w)

    #     total_weight = sum(weights)
    #     if total_weight == 0:
    #         return float('inf'), float('inf')

    #     mean = sum(w * f for w, f in zip(weights, f_values)) / total_weight
    #     var = sum(w * (f - mean) ** 2 for w, f in zip(weights, f_values)) / total_weight
    #     return mean, var

    def estimate_testing_dataset(self):
        """
        Estimate the function on the testing dataset
        """
        mu = []
        var = []
        for i, x in enumerate(self._data_testing):
            f_mean_t, f_var_t = self.estimate_local_f_and_uncertainty(x.tolist(),
                                                                      self._f_testing,
                                                                      self._data_testing)
            mu.append(f_mean_t)
            var.append(f_var_t)
        return mu, var

    def acquisition_ei(self, x, f_values, best_fx):
        """
        Acquisition function
        """
        f_mean, f_var = self.estimate_local_f_and_uncertainty(x, f_values, self._data_training)
        f_mean_t, _ = self.estimate_testing_dataset()
        tw = TrustWorthiness(ref=f_mean_t, pred=self._f_testing)
        corr_mean = tw.kendalltau_b_fast()
        if f_var == 0 or math.isinf(f_var):
            return 0.0, f_mean, 0.0
        f_std = math.sqrt(f_var)
        z = (best_fx - f_mean) / f_std
        ei = (best_fx - f_mean) * self.normal_cdf(z) + f_std * self.normal_pdf(z)
        adjusted_ei = ei * corr_mean
        return adjusted_ei, f_mean, f_std

    @classmethod
    def kde_resample(cls, ks: List[KERNEL], weights: List[float],
                     base_point: List[float], dim_weights: List[float] = None):
        """
        Resample using inferred distributions by introduced KDEs for a given dataset 
        """
        # Select kernel based on weighted probability
        kernel = random.choices(ks, weights=weights)[0]
        if dim_weights is not None:
            noise = kernel._sample_noise()
            for i, w in enumerate(dim_weights):
                noise[i] *= w
        else:
            noise = kernel._sample_noise()
        return [b + n for b, n in zip(base_point, noise)]

    def resample_near_incumbents(self, n_samples=5, dim_weights=None, incumbents=None):
        """
        Resample in the vicinity of incumbent points
        """
        if incumbents is None:
            incumbents = self._data_training
        samples = []
        lower_bounds = np.array([x[0] for x in self.var_limits])
        upper_bounds = np.array([x[1] for x in self.var_limits])
        for _ in range(n_samples):
            base = incumbents[np.random.randint(len(incumbents))]
            sample = self.kde_resample(ks=self.kernel, weights=[self.kernels[v] for
                                                                v in self.kernels.keys()],
                                       base_point=base, dim_weights=dim_weights)
            if lower_bounds is not None and upper_bounds is not None:
                for i, _ in enumerate(sample):
                    sample[i] = np.clip(sample[i], lower_bounds[i], upper_bounds[i])[0]
            samples.append(np.array(sample))
        return np.array(samples)

    def adjusted_expected_improvement(self, x, model, f_best):
        """
        Adjusted expected improvement
        """
        mu = model.predict(x)
        sigma = model.uncertainty(x)
        z = (f_best - mu) / (sigma + 1e-8)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        true_vals = self._f_testing
        pred_vals = np.array([model.predict(p) for p in self._data_testing])
        tw = TrustWorthiness(ref=true_vals, pred=pred_vals)
        corr_mean = tw.kendalltau_b_fast()
        return ei * max(corr_mean, 0)

    def bayesian_optimization_ensemble(self, models: dict, name, n_iterations=50):
        """
        Run BO using ensemble of models
        """
        print(f"\n=== Running BO with model: {name} ===")

        x = self._data_training
        y = self._f_training
        x_best = []
        y_best = []

        ensemble = AdaptiveEnsemble(models)
        ensemble.fit(x, y)

        best_ei = -np.inf

        for iteration in range(n_iterations):
            n_inc = max(1, len(x))
            inc_idx = np.argsort(y)[:n_inc]
            incumbents = x[inc_idx]
            f_best = y[inc_idx[0]]

            # sens_vars: TrustWorthiness = TrustWorthiness(x, y)
            dim_weights = compute_dimension_relevance(X=x, y=y, top_k=n_inc)

            candidates = self.resample_near_incumbents(dim_weights=dim_weights,
                                                       incumbents=incumbents)
            acq_vals = np.array([
                self.adjusted_expected_improvement(xc, ensemble, f_best) for xc in candidates])

            best_x = candidates[np.argmax(acq_vals)]
            unc = ensemble.uncertainty(best_x)
            best_y = ensemble.predict(best_x)
            z = ensemble.predict_with_zscore(self._data_testing, self._f_testing)
            z_score = [abs(zi) for zi in z]
            # err_t = ensemble.calculate_testing_error(self._data_testing, self._f_testing)
            # or 1 < np.mean(np.array(z_score)) or np.mean(np.array(z_score)) < 0:
            if np.isnan(unc) or np.isinf(unc) \
                or (abs(unc/best_y) > 1
                    and sum(z_score)/len(z_score) > 2):
                continue
            # x = np.vstack([x, best_x])
            # y = np.append(y, best_y)
            # if best_y < f_best and not np.isnan(unc):
            x_best.append(best_x)
            y_best.append(best_y)

        return x_best, y_best

    def resample_with_ei(self, f_values, best_fx, ei_threshold=0.01,
                         num_samples=1, max_attempts=1000):
        """
        Resample using EI criterion
        """
        accepted = []
        attempts = 0
        f_est_best = np.inf
        while len(accepted) < num_samples and attempts < max_attempts:
            base = random.choice(self._data_training)
            noise = [random.gauss(0, self.bandwidth) for _ in range(self.n_d)]
            candidate = self.resample_near_incumbents()
            ei, f_est, f_std = self.acquisition_ei(candidate, f_values, best_fx)
            if ei >= ei_threshold and f_est <= f_est_best:
                accepted = (candidate, f_est, f_std, ei)
                f_est_best = f_est
            attempts += 1
        return accepted

    def resample(self, size: int = None, seed: int = None,
                 display: bool = False) -> np.ndarray:
        """
        Resample using BO and KDE models
        """
        best_fx = min(self._f_training)
        if display:
            print("Initial best f(x):", best_fx)
        bw = self.bandwidth

        models = [
            # KernelWeightedAverage(bandwidth=bw, kw_calculator=self._combined_kernel),
            KernelRidgeRegression(bandwidth=bw, kw_calculator=self._combined_kernel),
            # LocalPolynomialRegression(bandwidth=bw,kw_calculator=self._combined_kernel),
            # GPWithMixedKernel(bandwidth=bw, kw_calculator=self._combined_kernel),
            # MDNInspired(bandwidth=bw, kw_calculator=self._combined_kernel),
            KNNKernelWeighted(k=self._data_training.shape[0],
                              bandwidth=bw, kw_calculator=self._combined_kernel)
        ]

        new_samples, best_fx = \
            self.bayesian_optimization_ensemble(models=models,
                                                name=GPWithMixedKernel.__name__,
                                                n_iterations=size)
        if display:
            print("\n🔍 Final best observed f(x):", new_samples)
        return np.array(new_samples)

    def generate_samples(self, ns: int):
        pass

    def set_options(self):
        pass

    def utilities(self):
        pass

    def methods(self):
        pass


class TPE(Sampling):
    """
    Tree of Parzen estimators base class
    """

    def __init__(self, good_data: np.ndarray, good_f_values: np.ndarray,
                 bad_data: np.ndarray, bad_f_values: np.ndarray,
                 n_r: int, vlim: np.ndarray, kernel_type: dict = {"Gaussian": 1},
                 seed: int = 10000, gamma=0.25):
        self._msgs: List[List[str]] = []
        self.kernel_funcs: Dict = KernelFunctions().kfs
        self.kernel: List[KERNEL] = []
        self.kernels: Dict[str, float] = kernel_type
        self.good_data: np.ndarray = np.atleast_2d(np.asarray(good_data))
        self.bad_data: np.ndarray = np.atleast_2d(np.asarray(bad_data))
        self.n_s, self.n_d = self.good_data.shape
        self.n_r = n_r
        self.good_f_values = np.atleast_1d(np.asarray(good_f_values))
        self.bad_f_values = np.atleast_1d(np.asarray(bad_f_values))
        self.resampled_data = np.zeros((self.n_r, self.n_d))
        self.seed = seed
        self.var_limits = copy.deepcopy(vlim)
        self.gamma = gamma

    @abstractmethod
    def _rank_data_and_initialize_kernels(self):
        pass

    @abstractmethod
    def _suggest(self):
        pass

    @classmethod
    def _get_density(cls, kernels: List[KERNEL], candidate, samples_vecs):
        kdes = []
        for i, _ in enumerate(kernels):
            kde = []
            # if size is None:
            #     size = int(kernels[i]._ne)
            for s_vec in samples_vecs:
                kde.append(kernels[i].kf_multivar(
                    np.array([candidate[i] - s_vec[i]
                              for i in range(len(candidate))])))
            kdes.append(kde)

        # Average the densities for sampling
        combined_density = np.mean(kdes, axis=0)

        # Normalize the combined density
        # combined_density /= np.sum(combined_density)

        return np.sum(combined_density) / len(combined_density)

    @abstractmethod
    def _rank_data_and_initialize_kernels(self):
        pass

    def resample(self, size, seed, scale):
        """
        Resample using inferred distributions of the given dataset 
        """
        return self._suggest()

    @classmethod
    def kde_resample(cls, ks: List[KERNEL], weights: List[float], data: List[List[float]]):
        """
        Resample using inferred distributions by introduced KDEs for a given dataset 
        """
        base_point = random.choice(data)
        # Select kernel based on weighted probability
        kernel = random.choices(ks, weights=weights)[0]
        noise = kernel._sample_noise()
        return [b + n for b, n in zip(base_point, noise)]

    def generate_samples(self):
        pass

    def methods(self):
        pass

    def set_options(self):
        pass

    def utilities(self):
        pass

    # @abstractmethod
    # def run(self, iterations=30):
    #     pass


class BiTPE(TPE):
    """
    Binary tree of Parzen estimators
    """

    def __init__(self, good_data: np.ndarray, good_f_values: np.ndarray,
                 bad_data: np.ndarray, bad_f_values: np.ndarray,
                 kernel_type: Dict = {"Gaussian": 1}, n_r: int = 0,
                 vlim: np.ndarray = None, bw_method=TUNING_METHOD.MLCV.name,
                 seed=10000, weights: Any = None, h: List[float] = None,
                 gamma=0.25):
        super().__init__(good_data=good_data, good_f_values=good_f_values, bad_data=bad_data,
                         bad_f_values=bad_f_values,
                         kernel_type=kernel_type, n_r=n_r,
                         vlim=vlim, seed=seed, gamma=gamma)
        self._rank_data_and_initialize_kernels(weights, bw_method, h)

    def _rank_data_and_initialize_kernels(self, bw_method=TUNING_METHOD.SCOTT.name,
                                          h: List[float] = None, weights: Any = None):
        inc_idx = np.argsort(self.good_f_values)
        self.good_obs = self.good_data[inc_idx]
        inc_idx = np.argsort(self.bad_f_values)
        self.bad_obs = self.bad_data[inc_idx]
        self._weights = None
        self.good_kernel: List[KERNEL] = []
        self.bad_kernel: List[KERNEL] = []
        for k in self.kernels.keys():
            if k in self.kernel_funcs.keys():
                kf = self.kernel_funcs.get(k)
                self.good_kernel.append(kf(data=self.good_obs,
                                           weights=self._weights,
                                           bw_method=bw_method, h=h))
                self.good_kernel[-1]._calc_covariance()
                self.bad_kernel.append(kf(data=self.bad_obs,
                                          weights=self._weights,
                                          bw_method=bw_method, h=h))
            else:
                self._msgs.append([1, "Unknown kernel type. "
                                   "Switched to the default Gaussian kernel."])
                self.good_kernel.append(Gaussian(data=self.good_obs,
                                                 weights=self._weights,
                                                 bw_method=bw_method, h=h))
                self.bad_kernel.append(Gaussian(data=self.bad_obs,
                                                weights=self._weights,
                                                bw_method=bw_method, h=h))

    def _suggest(self):

        # Calculate bandwidths per dimension
        bw_good = [k.tune_bw() for k in self.good_kernel]
        bw_bad = [k.tune_bw() for k in self.bad_kernel]

        # 2. Sample candidates and maximize l(x)/g(x)
        best_ratio = 1
        best_candidate = []
        sc = 0
        candidates = []
        for _ in range(self.n_r):
            # Sample from the 'good' GMM (pick a point and add Gaussian noise)
            random.seed(self.seed+sc)
            sc += 1
            # base_vec = self.good_obs[0]
            if len(candidates) >= self.n_r:
                break
                # lb = v[0]
                # ub = v[1]
                # sigma = (ub - lb) / np.sqrt(len(self.good_obs))
                # val = random.normalvariate(base_vec[i],  sigma)
                # val = max(lb, min(ub, val))  # Clip
            candidates += \
                self.kde_resample(ks=self.good_kernel,
                                  weights=[self.kernels[k]
                                           for k in self.kernels.keys()],
                                  data=self.good_data)

            # Calculate densities
        for candidate_vec in candidates:
            l_x = 1
            g_x = 1
            # for i in range(len(self.good_obs)):
            l_x = self._get_density(self.good_kernel, candidate_vec, self.good_obs)
            g_x = self._get_density(self.bad_kernel, candidate_vec, self.bad_obs)

            # for i in range(v.shape[0]):
            #     l_x *= l_x_all[i]
            #     g_x *= g_x_all[i]

            ratio = np.mean(l_x) / max(np.mean(g_x), 1e-12)

            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate.append(candidate_vec)

        return np.array(best_candidate)


if __name__ == "__main__":
    """ Samplers library """
