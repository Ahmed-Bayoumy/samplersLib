# test_bayesian_active_sampling.py
import numpy as np
import pytest

# Import the class under test
from samplersLib.samplers import BayesianActiveSampling, KernelRidgeRegression


# ----------------------------------------------------------------------
# Helper fixtures
# ----------------------------------------------------------------------
@pytest.fixture
# pylint: disable=missing-function-docstring
def synthetic_data():
    """Create a tiny 2‑D dataset with a simple quadratic objective."""
    rng = np.random.RandomState(0)  # pylint: disable=no-member
    # 8 points in 2‑D
    X = rng.uniform(-1, 1, size=(8, 2))
    # f(x) = (x0‑0.2)^2 + (x1+0.3)^2  (minimum near (0.2, -0.3))
    y = np.sum((X - np.array([0.2, -0.3])) ** 2, axis=1)
    # variable limits for each dimension
    vlim = np.array([[-1, 1], [-1, 1]])
    return X, y, vlim


# ----------------------------------------------------------------------
# Basic construction & internal split sanity checks
# ----------------------------------------------------------------------
# pylint: disable=missing-function-docstring
def test_instantiation_and_split(synthetic_data):
    X, y, vlim = synthetic_data
    sampler = BayesianActiveSampling(
        data=X,
        f_values=y,
        n_r=5,
        vlim=vlim,
        kernel_type={"Gaussian": 1},
        bw_method="scott",
        seed=42,
        weights=None,
        h=[0.2],
    )

    # internal attributes should be set
    assert sampler.n_s == X.shape[0]
    assert sampler.n_d == X.shape[1]

    # training set must contain at least one point (the last point is forced in)
    assert sampler._data_training.shape[0] >= 1
    # testing set should be non‑empty for this tiny example
    assert sampler._data_testing.shape[0] > 0

    # covariance matrix must be square and match dimensionality
    assert sampler.cov.shape == (sampler.n_d, sampler.n_d)


# ----------------------------------------------------------------------
# Kernel combination returns a scalar weight
# ----------------------------------------------------------------------
# pylint: disable=missing-function-docstring
def test_combined_kernel_returns_scalar(synthetic_data):
    X, y, vlim = synthetic_data
    sampler = BayesianActiveSampling(
        data=X,
        f_values=y,
        n_r=5,
        vlim=vlim,
        kernel_type={"Gaussian": 1},
        bw_method="scott",
        seed=0,
        h=[0.1],
    )

    x = X[0]
    xi = X[1]
    w = sampler._combined_kernel(x, xi)

    # weight must be a real number (float) and non‑negative
    assert isinstance(w, float)
    assert w >= 0.0


# ----------------------------------------------------------------------
# Local estimate returns mean & variance of the correct shape
# ----------------------------------------------------------------------
# pylint: disable=missing-function-docstring
def test_estimate_local_f_and_uncertainty(synthetic_data):
    X, y, vlim = synthetic_data
    sampler = BayesianActiveSampling(
        data=X,
        f_values=y,
        n_r=5,
        vlim=vlim,
        kernel_type={"Gaussian": 1},
        bw_method="scott",
        seed=1,
        h=[0.15],
    )

    # pick a random query point inside the domain
    query = np.array([0.0, 0.0])
    mean, var = sampler.estimate_local_f_and_uncertainty(query, y, X)

    assert isinstance(mean, float)
    assert isinstance(var, float)
    # variance should be non‑negative
    assert var >= 0.0


# ----------------------------------------------------------------------
# Acquisition EI returns a tuple (ei, mean, std) with sensible values
# ----------------------------------------------------------------------
# pylint: disable=missing-function-docstring
def test_acquisition_ei(synthetic_data):
    X, y, vlim = synthetic_data
    sampler = BayesianActiveSampling(
        data=X,
        f_values=y,
        n_r=5,
        vlim=vlim,
        kernel_type={"Gaussian": 1},
        bw_method="scott",
        seed=2,
        h=[0.2],
    )

    best_fx = np.min(y)
    ei, f_mean, f_std = sampler.acquisition_ei(X[0], y, best_fx)

    assert isinstance(ei, float)
    assert isinstance(f_mean, float)
    assert isinstance(f_std, float)
    # EI should be non‑negative
    assert ei >= 0.0
    # Standard deviation must be non‑negative
    assert f_std >= 0.0


# ----------------------------------------------------------------------
# Resample (high‑level BO loop) returns an array of new points
# ----------------------------------------------------------------------
# pylint: disable=missing-function-docstring
def test_resample_returns_samples(synthetic_data):
    X, y, vlim = synthetic_data
    sampler = BayesianActiveSampling(
        data=X,
        f_values=y,
        n_r=5,
        vlim=vlim,
        kernel_type={"Gaussian": 1},
        bw_method="scott",
        seed=3,
        h=[0.25],
    )

    # ask for 3 new samples; display=False keeps output quiet
    new_samples = sampler.resample(size=3, display=False)

    # result should be a NumPy array of shape (n_samples, n_dim)
    assert isinstance(new_samples, np.ndarray)
    assert new_samples.shape == (3, X.shape[1])

    # each new point must respect the variable limits
    lower, upper = vlim[:, 0], vlim[:, 1]
    assert np.all(new_samples >= lower) and np.all(new_samples <= upper)


# ----------------------------------------------------------------------
# Optional: sanity check that the ensemble BO loop does not raise errors
# ----------------------------------------------------------------------
# pylint: disable=missing-function-docstring
def test_bayesian_optimization_ensemble_runs(synthetic_data):
    X, y, vlim = synthetic_data
    sampler = BayesianActiveSampling(
        data=X,
        f_values=y,
        n_r=5,
        vlim=vlim,
        kernel_type={"Gaussian": 1},
        bw_method="scott",
        seed=4,
        h=[0.2],
    )

    # Use a single simple model (KernelRidgeRegression) to keep the test fast
    model = KernelRidgeRegression(bandwidth=[0.2], kw_calculator=sampler._combined_kernel)

    # The method should return two lists (incumbents & their values) without error
    x_best, y_best = sampler.bayesian_optimization_ensemble(models=[model], name="kr", n_iterations=5)
    assert isinstance(x_best, list)
    assert isinstance(y_best, list)
    # lengths may be zero (no improvement) – just ensure they match
    assert len(x_best) == len(y_best)
