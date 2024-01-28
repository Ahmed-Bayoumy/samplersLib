from typing import Dict, Any, Protocol, List
import copy
from dataclasses import dataclass
from pyDOE2 import lhs
import numpy as np
from scipy.spatial.distance import cdist, pdist
from .kernels import kernel as KERNEL, Gaussian, Epanechnikov, cosine, linear, uniformRectangular, triweight, tricube, Silverman, Sigmoid, biweight, logistic
from ._common import *

@dataclass
class sampling(Protocol):
  """
    Sampling methods template
  """
  _ns: int
  _varLimits: np.ndarray
  _options: Dict[str, Any]

  @property
  def n_s(self):
    return self._ns

  @n_s.setter
  def n_s(self, value: int) -> int:
    self._ns = value

  @property
  def varLimits(self):
    return self._varLimits

  @varLimits.setter
  def varLimits(self, value: np.ndarray) -> np.ndarray:
    self._varLimits = copy.deepcopy(value)

  @property
  def options(self):
    return self._options

  @options.setter
  def options(self, value: Dict[str, Any]) -> Dict[str, Any]:
    self._options = copy.deepcopy(value)

  def scale_to_limits(self, S: np.ndarray) -> np.ndarray:
    """
      Scale the samples from the unit hypercube to the specified limit.
    """
    n = self.varLimits.shape[0]
    for i in range(n):
      S[:, i] = self.varLimits[i, 0] + S[:, i] * \
            (self.varLimits[i, 1] - self.varLimits[i, 0])
      if "msize" in self.options.keys():
        s = S[:, i]
        nr = int((self.varLimits[i, 1] - self.varLimits[i, 0])/self.options["msize"])
        mod = (s % ((self.varLimits[i, 1] - self.varLimits[i, 0])/nr))
        S[:, i] = s - mod
        
    return S

  def generate_samples(self, ns: int):
    """ Compute the requested number of sampling points.
      The number of dimensions (nx) is determined based on `varLimits.shape[0].` """
    ...

  def set_options(self):
    ...

  def utilities(self):
    ...

  def methods(self):
    ...

@dataclass
class FullFactorial(sampling):
  def __init__(self, ns: int, w: np.ndarray, c: bool, vlim: np.ndarray):
    self.options = {}
    self.options["weights"] = copy.deepcopy(w)
    self.options["clip"] = c
    self.varLimits = copy.deepcopy(vlim)
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
    nx = self.varLimits.shape[0]

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

@dataclass
class LHS(sampling):
  def __init__(self, ns: int, vlim: np.ndarray):
    self.options = {}
    self.options["criterion"] = "ExactSE"
    self.options["randomness"] = 10000
    self.n_s = ns
    self.varLimits = copy.deepcopy(vlim)

  def utilities(self):
    pass

  def set_options(self, c: str, r: Any):
    self.options["criterion"] = c
    self.options["randomness"] = r

  def generate_samples(self):
    nx = self.varLimits.shape[0]

    if isinstance(self.options["randomness"], np.random.RandomState):
      self.random_state = self.options["randomness"]
    elif isinstance(self.options["randomness"], int):
      self.random_state = np.random.RandomState(
          self.options["randomness"])
    else:
      self.random_state = np.random.RandomState()

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
      return self._ExactSE(nx, ns)

  def _optimizeExactSE(self, X, T0=None,
                       outer_loop=None, inner_loop=None, J=20, tol=1e-3,
                       p=10, return_hist=False, fixed_index=[]):

    # Initialize parameters if not defined
    if T0 is None:
      T0 = 0.005 * self._phi_p(X, p=p)
    if inner_loop is None:
      inner_loop = min(20 * X.shape[1], 100)
    if outer_loop is None:
      outer_loop = min(int(1.5 * X.shape[1]), 30)

    T = T0
    X_ = X[:]  # copy of initial plan
    X_best = X_[:]
    d = X.shape[1]
    PhiP_ = self._phi_p(X_best, p=p)
    PhiP_best = PhiP_

    hist_T = list()
    hist_proba = list()
    hist_PhiP = list()
    hist_PhiP.append(PhiP_best)

    # Outer loop
    for z in range(outer_loop):
      PhiP_oldbest = PhiP_best
      n_acpt = 0
      n_imp = 0
      # Inner loop
      for i in range(inner_loop):
        modulo = (i + 1) % d
        l_X = list()
        l_PhiP = list()
        for j in range(J):
          l_X.append(X_.copy())
          l_PhiP.append(self._phi_p_transfer(l_X[j], k=modulo, phi_p=PhiP_, p=p, fixed_index=fixed_index))
        l_PhiP = np.asarray(l_PhiP)
        k = np.argmin(l_PhiP)
        PhiP_try = l_PhiP[k]
        # Threshold of acceptance
        if PhiP_try - PhiP_ <= T * self.random_state.rand(1)[0]:
          PhiP_ = PhiP_try
          n_acpt = n_acpt + 1
          X_ = l_X[k]
          # Best plan retained
          if PhiP_ < PhiP_best:
            X_best = X_
            PhiP_best = PhiP_
            n_imp = n_imp + 1
        hist_PhiP.append(PhiP_best)

      p_accpt = float(n_acpt) / inner_loop  # probability of acceptance
      p_imp = float(n_imp) / inner_loop  # probability of improvement

      hist_T.extend(inner_loop * [T])
      hist_proba.extend(inner_loop * [p_accpt])

    if PhiP_best - PhiP_oldbest < tol:
      # flag_imp = 1
      if p_accpt >= 0.1 and p_imp < p_accpt:
        T = 0.8 * T
      elif p_accpt >= 0.1 and p_imp == p_accpt:
        pass
      else:
        T = T / 0.8
    else:
      # flag_imp = 0
      if p_accpt <= 0.1:
        T = T / 0.7
      else:
        T = 0.9 * T

    hist = {"PhiP": hist_PhiP, "T": hist_T, "proba": hist_proba}

    if return_hist:
      return X_best, hist
    else:
      return X_best

  def _phi_p(self, X, p=10):

    return ((pdist(X) ** (-p)).sum()) ** (1.0 / p)

  def _phi_p_transfer(self, X, k, phi_p, p, fixed_index):
    """ Optimize how we calculate the phi_p criterion. """

    # Choose two (different) random rows to perform the exchange
    i1 = self.random_state.randint(X.shape[0])
    while i1 in fixed_index:
      i1 = self.random_state.randint(X.shape[0])

    i2 = self.random_state.randint(X.shape[0])
    while i2 == i1 or i2 in fixed_index:
      i2 = self.random_state.randint(X.shape[0])

    X_ = np.delete(X, [i1, i2], axis=0)

    dist1 = cdist([X[i1, :]], X_)
    dist2 = cdist([X[i2, :]], X_)
    d1 = np.sqrt(
        dist1 ** 2 + (X[i2, k] - X_[:, k]) ** 2 - (X[i1, k] - X_[:, k]) ** 2
    )
    d2 = np.sqrt(
        dist2 ** 2 - (X[i2, k] - X_[:, k]) ** 2 + (X[i1, k] - X_[:, k]) ** 2
    )

    res = (phi_p ** p + (d1 ** (-p) - dist1 ** (-p) + d2 ** (-p) - dist2 ** (-p)).sum()) ** (1.0 / p)
    X[i1, k], X[i2, k] = X[i2, k], X[i1, k]

    return res

  def _ExactSE(self, dim, nt, fixed_index=[], P0=[]):
    # Parameters of Optimize Exact Solution Evaluation procedure
    if len(fixed_index) == 0:
      P0 = lhs(dim, nt, criterion=None, random_state=self.random_state)
    else:
      P0 = P0
      self.random_state = np.random.RandomState()
    J = 20
    outer_loop = min(int(1.5 * dim), 30)
    inner_loop = min(20 * dim, 100)

    P, _ = self._optimizeExactSE(
        P0,
        outer_loop=outer_loop,
        inner_loop=inner_loop,
        J=J,
        tol=1e-3,
        p=10,
        return_hist=True,
        fixed_index=fixed_index,
    )
    return P

  def expand_lhs(self, x, n_points, method="basic"):
    varLimits = self.options["varLimits"] if self.varLimits is None else self.varLimits

    new_num = len(x) + n_points

    # Evenly spaced intervals with the final dimension of the LHS
    intervals = []
    for i in range(len(varLimits)):
      intervals.append(np.linspace(
          varLimits[i][0], varLimits[i][1], new_num + 1))

    # Creates a subspace with the rows and columns that have no points
    # in the new space
    subspace_limits = [[]] * len(varLimits)
    subspace_bool = []
    for i in range(len(varLimits)):
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
    sampling_new = LHS(ns=n_points, vlim=np.array([[0.0, 1.0]] * len(varLimits)))
    x_subspace = sampling_new.generate_samples()

    column_index = 0
    sorted_arr = x_subspace[x_subspace[:, column_index].argsort()]

    for j in range(len(varLimits)):
      for i in range(len(sorted_arr)):
        sorted_arr[i, j] = subspace_limits[j][i][0] + sorted_arr[i, j] * (
            subspace_limits[j][i][1] - subspace_limits[j][i][0]
        )

    H = np.zeros_like(sorted_arr)
    for j in range(len(varLimits)):
      order = np.random.permutation(len(sorted_arr))
      H[:, j] = sorted_arr[order, j]

    x_new = np.concatenate((x, H), axis=0)

    if method == "ExactSE":
      # Sampling of the new subspace
      sampling_new = LHS(ns=n_points, vlim=varLimits)
      x_new = sampling_new._ExactSE(
          len(x_new), len(x_new), fixed_index=np.arange(0, len(x), 1), P0=x_new
      )

    return x_new

@dataclass
class RS(sampling):
  def __init__(self, ns: int, vlim: np.ndarray, options: Dict[str, Any] = {}):
    self.options = options
    self.n_s = ns
    self.varLimits = copy.deepcopy(vlim)

  def generate_samples(self):
    nx = self.varLimits.shape[0]
    if self.options != {} and "randomness" in self.options:
      np.random.seed(self.options["randomness"])
    return self.scale_to_limits(np.random.rand(self.n_s, nx))

  def methods(self):
    pass

  def utilities(self):
    pass

  def set_options(self):
    pass

@dataclass
class halton(sampling):
  def __init__(self, ns: int, vlim: np.ndarray, is_ham: bool = True):
    self.options = {}
    self.n_s = ns
    self.varLimits = copy.deepcopy(vlim)
    self.ishammersley = is_ham

  def prime_generator(self, n: int):
    prime_list = []
    current_no = 2
    while len(prime_list) < n:
      for i in range(2, current_no):
        if (current_no % i) == 0:
            break
      else:
        prime_list.append(current_no)
      current_no += 1
    return prime_list
  
  def base_conv(self, a, b):
    string_representation = []
    if a < b:
      string_representation.append(str(a))
    else:
      while a > 0:
        a, c = (a // b, a % b)
        string_representation.append(str(c))
      string_representation = (string_representation[::-1])
    return string_representation
  
  def data_sequencing(self, pb):
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
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool8)
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
      no_features = self.varLimits.shape[0]
      # Generate list of no_features prime numbers
      prime_list = self.prime_generator(no_features)
      sample = np.zeros((self.n_s, no_features))
      for i in range(0, no_features):
        sample[:, i] = self.data_sequencing(prime_list[i])
      # Scale input data, then find data points closest in sample space. Unscale before returning points
      min_ = np.min(self.varLimits, axis=1)
      max_ = np.max(self.varLimits, axis=1)
      sample = sample * (max_ - min_) + min_
    else:
      big_number = 10
      dim = self.varLimits.shape[0]
      while 'Not enought primes':
          base = self.primes_from_2_to(big_number)[:dim]
          if len(base) == dim:
              break
          big_number += 1000

      # Generate a sample using a Van der Corput sequence per dimension.
      sample = [self.van_der_corput(self.n_s + 1, dim) for dim in base]
      sample = np.stack(sample, axis=-1)[1:]
      min_ = np.min(self.varLimits, axis=1)
      max_ = np.max(self.varLimits, axis=1)
      sample = sample * (max_ - min_) + min_

    return sample

  def methods(self):
    pass

  def utilities(self):
    pass

  def set_options(self):
    pass

@dataclass
class activeSampling(sampling):
  kernel: KERNEL = None
  n_r: int = 0
  data: np.ndarray = None
  resampled_data: np.ndarray = None
  n_d: int = 0
  n_s: int = 0
  _weights: Any = None
  _msgs: List[List[str]] = None
  _ne: int = 0
  
  def __init__(self, data: np.ndarray, n_r: int, vlim: np.ndarray, kernel_type: str = "Gaussian", bw_method = TUNING_METHOD.SCOTT.name, seed: int = 10000, weights: Any = None):
    self.data = np.atleast_2d(np.asarray(data))
    self._msgs = []
    if self.data.size <= 1:
      self._msgs.append([2, "`data` passed in to the `activeSampling` constructor should include multiple sample points."])
      raise ValueError("`data` passed in to the `activeSampling` constructor should include multiple sample points.")
    self.n_r = n_r
    self.n_s, self.n_d = self.data.shape
    if weights is not None:
      self._weights = np.atleast_1d(weights).astype(float)
      self._weights /= sum(self._weights)
      if not self._weights.ndim == 1:
        self._msgs.append([2, "`weights` passed in to the `activeSampling` constructor should be on-diemsional vector."])
      if not len(self._weights) == self.n_s:
        self._msgs.append([2, "`weights` passed in to the `activeSampling` constructor should have the same size of the input `data` array."])
        raise ValueError("`weights` passed in to the `activeSampling` constructor should have the same size of the input `data` array.")
      self._ne = 1/sum(self._weights**2)


    self.resampled_data = np.zeros((self.n_r, self.n_d))
    self.seed = seed
    self.varLimits = copy.deepcopy(vlim)
    self.data_normalized = self.data / (vlim[:, 1] - vlim[:, 0])

    if kernel_type == "linear" or kernel_type == "triangular":
      self.kernel = linear(data=self.data, weights=self._weights, bw_method=bw_method)
    elif kernel_type == "Gaussian":
      self.kernel = Gaussian(data=data, vlim=vlim, weights=self._weights, bw_method=bw_method, n_r=self.n_r)
    elif kernel_type == "Epanechnikov":
      self.kernel = Epanechnikov(data=self.data, weights=self._weights, bw_method=bw_method)
    elif kernel_type == "cosine":
      self.kernel = cosine(data=self.data, weights=self._weights, bw_method=bw_method)
    elif kernel_type == "uniformRectangular":
      self.kernel = uniformRectangular(data=self.data, weights=self._weights, bw_method=bw_method)
    elif kernel_type == "triweight":
      self.kernel = triweight(data=self.data, weights=self._weights, bw_method=bw_method)
    elif kernel_type == "tricube":
      self.kernel = tricube(data=self.data, weights=self._weights, bw_method=bw_method)
    elif kernel_type == "Silverman":
      self.kernel = Silverman(data=self.data, weights=self._weights, bw_method=bw_method)
    elif kernel_type == "Sigmoid":
      self.kernel = Sigmoid(data=self.data, weights=self._weights, bw_method=bw_method)
    elif kernel_type == "biweight":
      self.kernel = biweight(data=self.data, weights=self._weights, bw_method=bw_method)
    elif kernel_type == "logistic":
      self.kernel = logistic(data=self.data, weights=self._weights, bw_method=bw_method)
    else:
      self._msgs([1, "Unknown kernel type. Switched to the default Gaussian kernel."])
      self.kernel = Gaussian(data=self.data, weights=self._weights, bw_method=bw_method)

  def resample(self, size=None, seed=None):
    if size is None:
      size = int(self.kernel._ne)
    
    if size >self.kernel._points.shape[0]:
      size = self.kernel._points.shape[0]
    
    if self.kernel.est_pdf is None:
      self.kernel.est_pdf = self.kernel.estimate_pdf()
    if np.any(np.isnan(self.kernel.est_pdf)):
      for i in range(len(self.kernel.est_pdf)):
        if np.isnan(self.kernel.est_pdf[i]):
          self.kernel.est_pdf[i] = 0
      if sum(self.kernel.est_pdf) <= 0:
        self.kernel.est_pdf = np.atleast_1d([1/len(self.kernel.est_pdf)]*len(self.kernel.est_pdf))
      else:
        self.kernel.est_pdf /= sum(self.kernel.est_pdf)
    
    random_state = np.random.RandomState(seed)
    if type(self.kernel).__name__ == "Gaussian" and self.kernel._cov is not None:
      normDist = np.transpose(random_state.multivariate_normal(
          np.zeros((self.n_d,), float), self.kernel._cov, size=size
      ))
      indices = random_state.choice(self.kernel._points.shape[0], size=size, p=self.kernel.est_pdf)
      means = self.data[indices, :]
      new = means + normDist.T
    else:
      indices1 = random_state.choice(self.kernel.data.shape[0], size=size)
      indices2 = random_state.choice(self.kernel._points.shape[0], size=size, p=abs(self.kernel.est_pdf))
      means1 = self.data[indices1, :]
      means2 = self.kernel._points[indices2, :]
      new = (means1 + means2)/2
    
    omit = []
    for i in range(size):
      for j in range(self.n_d):
        if new[i,j] < self.varLimits[j, 0] or new[i,j] > self.varLimits[j, 1]:
          omit.append(i)
    
    return np.delete(new, omit, axis=0)


if __name__ == "__main__":
  """ Samplers library """