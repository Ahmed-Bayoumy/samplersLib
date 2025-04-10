import importlib
from samplersLib import samplers, kernels
moduleName= 'matplotlib'
if importlib.import_module(moduleName):
  from matplotlib import pyplot as plt

import numpy as np
from scipy import stats
from multiprocessing import freeze_support


rand_seed = 100

def test_combined_kernels(plotting = False):
  D, _ = make_data_uniform(data_count=200)
  data = np.atleast_2d(D)
  Ker1 = kernels.Laplace(data=data.T, vlim=np.atleast_1d([-1, 1]), bw_method=kernels.TUNING_METHOD.MLCV.name)
  Ker2 = kernels.Epanechnikov(data=data.T, vlim=np.atleast_1d([-1, 1]), bw_method=kernels.TUNING_METHOD.MLCV.name)
  Ker3 = kernels.Gaussian(data=data.T, vlim=np.atleast_1d([-1, 1]), bw_method=kernels.TUNING_METHOD.MLCV.name)


  # plot x_norm and kde
  x = Ker1.x
  Ker1._generate_nd_grid()
  Ker1.tune_bw()
  # Ker.h[0] = 10
  kde_norm1 = Ker1.calculate()
  kde1 = Ker1.estimate_pdf()

  # plot x_norm and kde
  x = Ker2.x
  Ker2._generate_nd_grid()
  Ker2.tune_bw()
  # Ker.h[0] = 10
  kde_norm2 = Ker2.calculate()
  kde2 = Ker2.estimate_pdf()

  # plot x_norm and kde
  x = Ker3.x
  Ker3._generate_nd_grid()
  Ker3.tune_bw()
  # Ker.h[0] = 10
  kde_norm3 = Ker3.calculate()
  kde3 = Ker3.estimate_pdf()

  kde_norm = [kde1, kde2, kde3]

  # Average the densities for sampling
  combined_density = np.mean(kde_norm, axis=0)
  

  # Sample from the combined density
  x_range = np.linspace(min(D) - 1, max(D) + 1, 200)
  samples = np.random.choice(x_range, size=200, p=combined_density/np.sum(combined_density))


  if importlib.import_module('matplotlib'):
    plotting = True
  if plotting:
    # Create two subplots and unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(Ker1._points, kde_norm1, label='Laplace')
    ax1.plot(Ker2._points, kde_norm2, label='Epanechnikov')
    ax1.plot(Ker3._points, kde_norm3, label='Gaussian')
    ax1.hist(data.T, density=True, alpha=0.2, label='Histogram', bins=30, rwidth=0.9)
    ax1.plot(data.T, np.full_like(data.T, -0.02), '|k', markeredgewidth=1)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-0.02, 2])
    ax1.legend()

  # p = np.zeros(Ker._points.shape[0])
  # for i in range(1,kde_norm.shape[0]):
  #   p[i] = (0.5*(kde_norm[i] + kde_norm[i-1]))

  # p /= sum(p)
  # indices1 = np.random.choice(data.T.shape[0], size=data.T.shape[0])
  # indices2 = np.random.choice(Ker._points.shape[0], size=Ker._points.shape[0], p=p)
  # means1 = data.T[indices1, :]
  # means2 = Ker._points[indices2, :]
  # resampled_data = (means1 + means2)
  if plotting:
    ax2.plot(Ker1._points, kde_norm1, label='Laplace')
    ax2.plot(Ker2._points, kde_norm2, label='Epanechnikov')
    ax2.plot(Ker3._points, kde_norm3, label='Gaussian')
    ax2.hist(samples, density=True, alpha=0.2, label='Histogram', bins=30, rwidth=0.9)
    ax2.plot(samples, np.full_like(samples, -0.02), '|k', markeredgewidth=1)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-0.02, 2])
    ax2.legend()
    plt.show()
  
  return kde_norm

def test_kernels(plotting = False):
  D, _ = make_data_uniform(data_count=200)
  data = np.atleast_2d(D)
  Ker = kernels.Gaussian_RBF(data=data.T, vlim=np.atleast_1d([-1, 1]), bw_method=kernels.TUNING_METHOD.SCOTT.name)

  # plot x_norm and kde
  x = Ker.x
  Ker._generate_nd_grid()
  # Ker.tune_bw()
  # Ker.h[0] = 10
  kde_norm = Ker.calculate(1)
  if not importlib.import_module(moduleName):
    plotting = False
  if plotting:
    # Create two subplots and unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(Ker._points, kde_norm, label='kde')
    ax1.hist(data.T, density=True, alpha=0.2, label='Histogram', bins=30, rwidth=0.9)
    ax1.plot(data.T, np.full_like(data.T, -0.02), '|k', markeredgewidth=1)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-0.02, 2])
    ax1.legend()

  p = Ker.estimate_pdf()

  p /= sum(p)
  indices1 = np.random.choice(data.T.shape[0], size=data.T.shape[0])
  indices2 = np.random.choice(Ker._points.shape[0], size=Ker._points.shape[0], p=p)
  means1 = data.T[indices1, :]
  means2 = Ker._points[indices2, :]
  resampled_data = (means1 + means2)
  if plotting:
    ax2.plot(Ker._points, kde_norm, label='kde')
    ax2.hist(resampled_data, density=True, alpha=0.2, label='Histogram', bins=30, rwidth=0.9)
    ax2.plot(resampled_data, np.full_like(resampled_data, -0.02), '|k', markeredgewidth=1)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-0.02, 2])
    ax2.legend()
    plt.show()
  
  return kde_norm, p

def make_data_uniform(data_count=100):
    alpha = 0.3
    np.random.seed(rand_seed)
    x = np.concatenate([
        np.random.uniform(-1, 1, int(data_count * alpha)),
        np.random.uniform(0, 1, int(data_count * (1 - alpha)))
    ])
    dist = lambda z: alpha * stats.uniform(-1, 1).pdf(z) + (1 - alpha) * stats.uniform(0, 1).pdf(z)
    return x, dist

def test_multivars(plotting = False):
  v = np.array([[-5.0, 10.0], [0.0, 15.0], [0.0, 15.0]])
  n = 300
  data = samplers.halton(ns=n, vlim=v).generate_samples()
  X, Y, Z = np.mgrid[-5:10:5j, 0:15:5j, 0:15:5j]
  positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

  AS = samplers.activeSampling(data=data, n_r=100, vlim=v, kernel_type="Gaussian") 
  # AS.kernel.h = np.atleast_1d([0.1, 1])
  AS.kernel.bw_method = "SILVERMAN"
  temp = AS.kernel.estimate_pdf(positions.T)
  ZZ = np.reshape(temp.T, X.shape)
  # S = activeSampling(data=data, n_r=10, vlim=v) 
  S = AS.resample(100, 10000)
  if importlib.import_module('matplotlib'):
    plotting = False

  if plotting:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    plt.contourf(X[:, :, 4], Y[:, :, 4], ZZ[:, :, 4], 100, cmap=plt.cm.YlGnBu)
    ax.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
    ax.plot(S[:, 0], S[:, 1], 'r.', markersize=2)
    # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    plt.show()

def measure(n):
  "Measurement model, return two coupled measurements."
  m1 = np.random.normal(size=n)
  m2 = np.random.normal(scale=0.5, size=n)
  return m1+m2, m1-m2

def test_bivariate(plotting = False):
  m1, m2 = measure(5000)
  xmin = m1.min()
  xmax = m1.max()
  ymin = m2.min()
  ymax = m2.max()

  X, Y = np.mgrid[xmin:xmax:10j, ymin:ymax:10j]
  positions = np.vstack([X.ravel(), Y.ravel()])
  values = np.vstack([m1, m2])

  AS = samplers.activeSampling(data=values.T, n_r=2025, vlim=np.array([[xmin, xmax], [ymin, ymax]]), kernel_type="Sigmoid") 
  AS.kernel.bw_method = "Silverman"
  temp = AS.kernel.estimate_pdf(positions.T)
  Z = np.reshape(temp.T, X.shape)

  S = AS.resample(20, 10000)
  if importlib.import_module('matplotlib'):
    plotting = False
  if plotting:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.contourf(X, Y, Z, 100, cmap=plt.cm.YlGnBu)
    ax.plot(m1, m2, 'k.', markersize=2)
    ax.plot(S[:, 0], S[:, 1], 'r.', markersize=2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.show()

def test_PSS(plotting = False):
  v = np.array([[-5.0, 10.0], [0.0, 15.0], [0.0, 15.0]])
  n = 300
  data = samplers.halton(ns=n, vlim=v).generate_samples()

  X, Y, Z = np.mgrid[-5:10:5j, 0:15:5j, 0:15:5j]
  positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

  PSS = samplers.TunablePSS(data=data, n_r=2025, vlim=v, seed=10000, num_particles=100, max_iter=100)
  samples = PSS.resample(1000, seed=12345)

  if plotting:
    # Plot pairs of dimensions
    for i in range(4):
      for j in range(i + 1, 3):
        plt.subplot(3, 3, i * 3 + j)
        plt.scatter(samples[:, i], samples[:, j], alpha=0.5, color='orange')
        plt.title(f'Samples: Dimension {i+1} vs Dimension {j+1}')
        plt.xlabel(f'X{i+1}')
        plt.ylabel(f'X{j+1}')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def test_SAS(plotting=False):
  # Parameters
  v = np.array([[-5.0, 10.0], [0.0, 15.0], [0.0, 15.0]])
  initial_solution = np.array([0.0, 0.0, 0.0])  # Starting point in 2D
  max_iter = 10000
  initial_temp = 1.0
  cooling_rate = 0.99 

  SAS = samplers.TunableSA(data=initial_solution, n_r=2025, vlim=v, seed=10000, max_iter=100, initial_temp=initial_temp, cooling_rate=cooling_rate)
  samples = SAS.resample(1000, seed=12345)

  if plotting:
    # Plot pairs of dimensions
    for i in range(4):
      for j in range(i + 1, 3):
        plt.subplot(3, 3, i * 3 + j)
        plt.scatter(samples[:, i], samples[:, j], alpha=0.5, color='orange')
        plt.title(f'Samples: Dimension {i+1} vs Dimension {j+1}')
        plt.xlabel(f'X{i+1}')
        plt.ylabel(f'X{j+1}')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
  """ Testing the samplers library """
  freeze_support()