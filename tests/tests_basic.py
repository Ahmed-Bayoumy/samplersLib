from samplersLib import samplers, kernels

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from multiprocessing import freeze_support


rand_seed = 100
def test_kernels(plotting = False):
  D, _ = make_data_uniform(data_count=200)
  data = np.atleast_2d(D)
  Ker = kernels.linear(data=data.T, vlim=np.atleast_1d([-1, 1]), bw_method=kernels.TUNING_METHOD.SCOTT.name)

  # plot x_norm and kde
  x = Ker.x
  Ker._generate_nd_grid()
  Ker.tune_bw()
  # Ker.h[0] = 10
  kde_norm = Ker.calculate()
  if plotting:
    # Create two subplots and unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(Ker._points, kde_norm, label='kde')
    ax1.hist(data.T, density=True, alpha=0.2, label='Histogram', bins=30, rwidth=0.9)
    ax1.plot(data.T, np.full_like(data.T, -0.02), '|k', markeredgewidth=1)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-0.02, 2])
    ax1.legend()

  p = np.zeros(Ker._points.shape[0])
  for i in range(1,kde_norm.shape[0]):
    p[i] = (0.5*(kde_norm[i] + kde_norm[i-1]))

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
  if plotting:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.contourf(X, Y, Z, 100, cmap=plt.cm.YlGnBu)
    ax.plot(m1, m2, 'k.', markersize=2)
    ax.plot(S[:, 0], S[:, 1], 'r.', markersize=2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.show()

if __name__ == "__main__":
  """ Testing the samplers library """
  freeze_support()