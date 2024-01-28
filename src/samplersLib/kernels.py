import numpy as np
from dataclasses import dataclass
from typing import Protocol, Any, Callable
import copy

from scipy import stats, optimize
from ._common import *

@dataclass
class kernel(Protocol):
  """ Protocol class for kernel functions """
  _ns: int = 0
  _nd: int = 0
  data: np.ndarray = None
  x: np.ndarray = None
  h: List[float] = None
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

  def _generate_nd_grid(self):
    v_edges: tuple = ()
    ranges: tuple =()
    pt = []
    for i in range(self._nd):
      pt.append(np.linspace(self.vlim[i, 0], self.vlim[i, 1], self._ne).T)

    self._points = np.vstack(np.meshgrid(*pt)).reshape(len(pt), -1).T
    self._points = self._points[:self._ne, :]

  def calculate(self, h: float = None):
    if h is None:
      bw = self.h
    else:
      self.h = h
      bw = h
    K = np.zeros(self._points.shape[0])
    for i in range(self._points.shape[0]):
      for j in range(self._ns):
        if self._nd == 1:
          K[i] += self.kf_univar((self._points[i]-self.data[j])/self.h)
        else:
          K[i] += self.kf_multivar((self._points[i]-self.data[j])/self.h)
    
      K[i] /= ((self._points.shape[0]-1))
    if h is not None:
      return -np.mean(np.log(K[K > 0]))
    else:
      return K
  
  def _opt(self):
    self.h = optimize.minimize(self.calculate, np.ones(self._nd)).x
    for i in range(self._nd):
      if self.h[i] > 10:
          self.bw_scott()

  def tune_bw(self):
    if self.bw_method == TUNING_METHOD.MLCV.name:
      self._opt()
    elif self.bw_method == TUNING_METHOD.SILVERMAN.name:
      self.bw_silverman()
    else:
      self.bw_scott()
    
    return self.h
  
  def set_bw(self, method):
    if method == TUNING_METHOD.MLCV.name:
      self._covariance_factor = self._opt
    elif method == TUNING_METHOD.SILVERMAN.name:
      self._covariance_factor = self._silverman
    else:
      self._covariance_factor = self._scott
    
    self._calc_covariance()
  
  def _scott(self):
    return np.power(self._ne, -1./(self._nd+4))
  
  def _silverman(self):
    return np.power(self._ne*(self._nd+2.0)/4.0, -1./(self._nd+4))
  
  def bounded(self, f, u):
    return f if np.abs(u) <= 1 else 0
  
  def bw_scott(self):
    std_dev = np.std(self.data, axis=0, ddof=1)
    n = len(self.data)
    self.h = [3.49 * std_dev * n ** (-0.333)]
    self.std_dev = std_dev

  def bw_silverman(self):
    sigma = self._select_sigma(self.data)
    n = len(self.data)
    self.h =  [0.9 * sigma * n ** (-0.2)]
    self.std_dev = np.sqrt(sigma)

  def _select_sigma(self, x):
    normalizer = 1.349
    iqr = (stats.scoreatpercentile(x, 75) - stats.scoreatpercentile(x, 25)) / normalizer
    std_dev = np.std(x, axis=0, ddof=1)
    return np.minimum(std_dev, iqr) if iqr > 0 else std_dev
  
  def _calc_covariance(self):
    """ Computes the covariance matrix for each kernel using the kernel BW (covariance factor). """
    try:
      self._factor = self._covariance_factor()

      if self._data_inv_cov == None:
        if self.weights is None:
          self.weights = np.ones(self._ns)/self._ns
        self._data_cov = np.atleast_2d(np.cov(self.data.T, 
                                                    rowvar=1, 
                                                    bias=True, 
                                                    aweights=self.weights))
        self._data_inv_cov = np.linalg.inv(self._data_cov)
      
      self._cov = self._data_cov * self._factor**2
      self.is_debugging = False
      if self.is_debugging:
        labs = [f'x{i}' for i in range(self._nd)]
        sns.heatmap(self._cov, annot=True, fmt='g', xticklabels=labs, yticklabels=labs)
        plt.show()
      self._inv_cov = self._data_inv_cov / self._factor**2
      # L = np.linalg.cholesky(self._cov*2*np.pi)
      # self._log_det = 2*np.log(np.diag(L)).sum()
    except:
      self._data_cov = None
      self._data_inv_cov = None
      self._cov = None
      self._inv_cov = None

  def kf_univar(self):
    ...
  
  def kf_multivar(self):
    ...

  def estimate_pdf(self, points=None):
    if points is None:
      if self._points is None:
        self._generate_nd_grid()
      self._points = np.atleast_2d(np.asarray(self._points))
    else:
      self._points = points
    
    return self.get_ke()
  
  def get_ke(self):
    if not isinstance(self.h, list) or self.h is None:
        self.tune_bw()
    self.est_pdf = np.atleast_1d(np.zeros((self._points.shape[0])))
    for i in range(self._ns):
      ei = np.atleast_1d(np.zeros((self._points.shape[0])))
      for j in range(self._points.shape[0]):
        z: np.ndarray = (self._points[j, :]-self.data[i, :])/(self.h if type(self).__name__ is not "Gaussian" or self._cov is None or (1/np.linalg.det(self._cov)) <= 1E-6 else np.ones(self._nd))
        ei[j] = self.kf_multivar(z)
      self.est_pdf += ei/self._points.shape[0]
    
    self.est_pdf = np.atleast_1d(abs(self.est_pdf)/sum(abs(self.est_pdf)))

    return np.asarray(self.est_pdf)

@dataclass
class Gaussian(kernel):
  def __init__(self, data:np.ndarray = None, vlim = None, res: int = 101, weights: Any = None, bw_method = TUNING_METHOD.SCOTT.name, point: np.ndarray = None, n_r: int = 0):
    self.data = copy.deepcopy(data)
    self._ns = data.shape[0]
    self._nd = data.shape[1]
    self.weights = weights
    if self.weights is not None:
      self._ne = 1/sum(self.weights**2)
    if n_r > 0:
      self._ne = n_r
    else:
      self._ne = self._ns
    if bw_method != TUNING_METHOD.MLCV.name:
      self.set_bw(bw_method)
    self.vlim = np.atleast_2d(vlim)
    self.bw_method = bw_method
    
    # self._generate_nd_grid()
  """ Gaussian kernel function """
  def kf_univar(self, u):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * u * u)
  
  def kf_multivar(self, z: np.ndarray):
    if self._cov is not None and (1/np.linalg.det(self._cov)) > 1E-6 and np.linalg.det(self._cov) > 1E-6:
      return (1.0 / (np.sqrt((2 * np.pi)**self._nd * np.linalg.det(self._cov))) * np.exp(-(np.linalg.solve(self._cov, z).T.dot(z)) / 2))
    else:
      return np.prod(1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * z * z))/np.prod(self.h)
  
@dataclass
class Epanechnikov(kernel):
  def __init__(self, data:np.ndarray = None, vlim = None, res: int = 101, weights: Any = None, bw_method = TUNING_METHOD.MLCV.name, point: np.ndarray = None):
    self.data = copy.deepcopy(data)
    self._ns = data.shape[0]
    self._nd = data.shape[1]
    self.weights = weights
    if self.weights is not None:
      self._ne = 1/sum(self.weights**2)
    self._ne = self._ns
    self.vlim = np.atleast_2d(vlim)
    self.bw_method = bw_method
  """ Epanechnikov kernel function """
  def kf_univar(self, u):
    return self.bounded((3 / 4 * (1 - u * u)), u)
  
  def kf_multivar(self, u):
    return self.bounded(np.prod(3 / 4 * (1 - u * u))/np.prod(self.h), np.prod(u))
  
@dataclass
class cosine(kernel):
  """ Cosine kernel function """
  def __init__(self, data:np.ndarray = None, vlim = None, res: int = 101, weights: Any = None, bw_method = TUNING_METHOD.MLCV.name, point: np.ndarray = None):
    self.data = copy.deepcopy(data)
    self._ns = data.shape[0]
    self._nd = data.shape[1]
    self.weights = weights
    if self.weights is not None:
      self._ne = 1/sum(self.weights**2)
    self._ne = self._ns
    self.vlim = np.atleast_2d(vlim)
    self.bw_method = bw_method

  def kf_univar(self, u):
    return self.bounded(np.pi / 4 * np.cos(np.pi / 2 * u), u)
  
  def kf_multivar(self, u):
    return self.bounded(np.prod(np.pi / 4 * np.cos(np.pi / 2 * u))/np.prod(self.h), np.prod(u))

@dataclass
class linear(kernel):
  """ Linear  kernel function """
  def __init__(self, data:np.ndarray = None, vlim = None, res: int = 101, weights: Any = None, bw_method = TUNING_METHOD.MLCV.name, point: np.ndarray = None):
    self.data = copy.deepcopy(data)
    self._ns = data.shape[0]
    self._nd = data.shape[1]
    self.weights = weights
    if self.weights is not None:
      self._ne = 1/sum(self.weights**2)
    self._ne = self._ns
    self.vlim = np.atleast_2d(vlim)
    self.bw_method = bw_method

  def kf_univar(self, u):
    return self.bounded(1 - np.abs(u), u)
  
  def kf_multivar(self, u):
    return self.bounded(np.prod(1 - np.abs(u))/np.prod(self.h), np.prod(u))

@dataclass
class uniformRectangular(kernel):
  """ Uniform rectangular kernel function """
  def __init__(self, data:np.ndarray = None, vlim = None, res: int = 101, weights: Any = None, bw_method = TUNING_METHOD.MLCV.name, point: np.ndarray = None):
    self.data = copy.deepcopy(data)
    self._ns = data.shape[0]
    self._nd = data.shape[1]
    self.weights = weights
    if self.weights is not None:
      self._ne = 1/sum(self.weights**2)
    self._ne = self._ns
    self.vlim = np.atleast_2d(vlim)
    self.bw_method = bw_method

  def kf_univar(self, u):
    return self.bounded(0.5, u)
  
  def kf_multivar(self, u):
    return self.bounded(0.5/np.prod(self.h), np.prod(u))

@dataclass
class triweight(kernel):
  """ Triweight kernel function """
  def __init__(self, data:np.ndarray = None, vlim = None, res: int = 101, weights: Any = None, bw_method = TUNING_METHOD.MLCV.name, point: np.ndarray = None):
    self.data = copy.deepcopy(data)
    self._ns = data.shape[0]
    self._nd = data.shape[1]
    self.weights = weights
    if self.weights is not None:
      self._ne = 1/sum(self.weights**2)
    self._ne = self._ns
    self.vlim = np.atleast_2d(vlim)
    self.bw_method = bw_method

  def kf_univar(self, u):
    return self.bounded((35/32)*(1-u**2)**3, u)
  
  def kf_multivar(self, u):
    return self.bounded(np.prod((35/32)*(1-u**2)**3)/np.prod(self.h), np.prod(u))

@dataclass
class tricube(kernel):
  """ tricube kernel function """
  def __init__(self, data:np.ndarray = None, vlim = None, res: int = 101, weights: Any = None, bw_method = TUNING_METHOD.MLCV.name, point: np.ndarray = None):
    self.data = copy.deepcopy(data)
    self._ns = data.shape[0]
    self._nd = data.shape[1]
    self.weights = weights
    if self.weights is not None:
      self._ne = 1/sum(self.weights**2)
    self._ne = self._ns
    self.vlim = np.atleast_2d(vlim)
    self.bw_method = bw_method

  def kf_univar(self, u):
    return self.bounded((70/81)*(1-np.abs(u)**3)**3, u)
  
  def kf_multivar(self, u):
    return self.bounded(np.prod((70/81)*(1-np.abs(u)**3)**3)/np.prod(self.h), np.prod(u))

@dataclass
class Silverman(kernel):
  """ Silverman kernel function """
  def __init__(self, data:np.ndarray = None, vlim = None, res: int = 101, weights: Any = None, bw_method = TUNING_METHOD.MLCV.name, point: np.ndarray = None):
    self.data = copy.deepcopy(data)
    self._ns = data.shape[0]
    self._nd = data.shape[1]
    self.weights = weights
    if self.weights is not None:
      self._ne = 1/sum(self.weights**2)
    self._ne = self._ns
    self.vlim = np.atleast_2d(vlim)
    self.bw_method = bw_method

  def kf_univar(self, u):
    return 0.5*np.exp(-(abs(u))/np.sqrt(2)) * np.sin((abs(u)/np.sqrt(2))+(np.pi/4))
  
  def kf_multivar(self, u):
    return self.bounded(np.prod(0.5*np.exp(-(abs(u))/np.sqrt(2)) * np.sin((abs(u)/np.sqrt(2))+(np.pi/4)))/np.prod(self.h), np.prod(u))

@dataclass
class Sigmoid(kernel):
  """ Sigmoid kernel function """
  def __init__(self, data:np.ndarray = None, vlim = None, res: int = 101, weights: Any = None, bw_method = TUNING_METHOD.MLCV.name, point: np.ndarray = None):
    self.data = copy.deepcopy(data)
    self._ns = data.shape[0]
    self._nd = data.shape[1]
    self.weights = weights
    if self.weights is not None:
      self._ne = 1/sum(self.weights**2)
    self._ne = self._ns
    self.vlim = np.atleast_2d(vlim)
    self.bw_method = bw_method

  def kf_univar(self, u):
    return (2/np.pi) *(1/(np.exp(u)+np.exp(-u)))
  
  def kf_multivar(self, u):
    return self.bounded(np.prod((2/np.pi) *(1/(np.exp(u)+np.exp(-u))))/np.prod(self.h), np.prod(u))

@dataclass
class biweight(kernel):
  """ biweight kernel function """
  def __init__(self, data:np.ndarray = None, vlim = None, res: int = 101, weights: Any = None, bw_method = TUNING_METHOD.MLCV.name, point: np.ndarray = None):
    self.data = copy.deepcopy(data)
    self._ns = data.shape[0]
    self._nd = data.shape[1]
    self.weights = weights
    if self.weights is not None:
      self._ne = 1/sum(self.weights**2)
    self._ne = self._ns
    self.vlim = np.atleast_2d(vlim)
    self.bw_method = bw_method

  def kf_univar(self, u):
    return self.bounded((15/16) *(1-u**2)**2, u)
  
  def kf_multivar(self, u):
    return self.bounded(np.prod((15/16) *(1-u**2)**2)/np.prod(self.h), np.prod(u))

@dataclass
class logistic(kernel):
  """ logistic kernel function """
  def __init__(self, data:np.ndarray = None, vlim = None, res: int = 101, weights: Any = None, bw_method = TUNING_METHOD.MLCV.name, point: np.ndarray = None):
    self.data = copy.deepcopy(data)
    self._ns = data.shape[0]
    self._nd = data.shape[1]
    self.weights = weights
    if self.weights is not None:
      self._ne = 1/sum(self.weights**2)
    self._ne = self._ns
    self.vlim = np.atleast_2d(vlim)
    self.bw_method = bw_method

  def kf_univar(self, u):
    return 1/((np.exp(u)+2+np.exp(-u)))
  
  def kf_multivar(self, u):
    return np.prod(1/((np.exp(u)+2+np.exp(-u))))/np.prod(self.h)






