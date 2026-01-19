import numpy as np
import pytest

# Import the class under test – adjust the import path if your package layout differs
from samplersLib.samplers import TUNING_METHOD, ActiveSampling


# ----------------------------------------------------------------------
# Helper fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def simple_data():
    """2‑D dataset (n_s=5, n_d=2) – no reduction path."""
    rng = np.random.default_rng(0)
    return rng.random((5, 2))


@pytest.fixture
def high_dim_data():
    """5‑D dataset (n_s=6, n_d=5) – triggers the reducer path."""
    rng = np.random.default_rng(1)
    return rng.random((6, 5))


@pytest.fixture
def var_limits():
    """Variable limits – one row per dimension, [lower, upper]."""
    # simple 0‑1 bounds for every dimension
    return np.column_stack((np.zeros(5), np.ones(5)))


# ----------------------------------------------------------------------
# Constructor tests
# ----------------------------------------------------------------------
def test_constructor_basic(simple_data):
    """Constructor should succeed with a valid 2‑D dataset."""
    vlim = np.column_stack((np.zeros(2), np.ones(2)))
    sampler = ActiveSampling(
        data=simple_data,
        n_r=2,
        vlim=vlim,
        kernel_type=["Gaussian"],
        bw_method=TUNING_METHOD.SCOTT.name,
        seed=123,
        h=[0.1] * 2,
    )
    # basic attributes are set
    assert sampler.n_s == simple_data.shape[0]
    assert sampler.n_d == simple_data.shape[1]
    assert sampler.kernel  # at least one kernel instance created


def test_constructor_invalid_data():
    """Passing a single point should raise a ValueError."""
    vlim = np.column_stack((np.zeros(2), np.ones(2)))
    with pytest.raises(ValueError):
        ActiveSampling(
            data=np.array([[0.5, 0.5]]),  # only one sample point
            n_r=2,
            vlim=vlim,
            kernel_type=["Gaussian"],
            bw_method=TUNING_METHOD.SCOTT.name,
            h=[0.1] * 2,
        )


def test_constructor_weights_mismatch(simple_data):
    """Weights length must match number of samples."""
    vlim = np.column_stack((np.zeros(2), np.ones(2)))
    # too few weights
    with pytest.raises(ValueError):
        ActiveSampling(
            data=simple_data,
            n_r=2,
            vlim=vlim,
            kernel_type=["Gaussian"],
            bw_method=TUNING_METHOD.SCOTT.name,
            weights=[0.5, 0.5],
            h=[0.1] * 2,  # length 2 vs n_s=5
        )


# ----------------------------------------------------------------------
# Standardisation / reduction tests
# ----------------------------------------------------------------------
def test_standardize_and_rd(simple_data):
    """standardize_data and rd should run and produce reduced data."""
    vlim = np.column_stack((np.zeros(2), np.ones(2)))
    sampler = ActiveSampling(
        data=simple_data,
        n_r=2,
        vlim=vlim,
        kernel_type=["Gaussian"],
        bw_method=TUNING_METHOD.SCOTT.name,
        h=[0.1] * 2,
    )
    sampler.standardize_data()
    # after standardisation we have zero‑mean, unit‑variance columns
    np.testing.assert_allclose(sampler.means, np.mean(sampler.data_scaled, axis=0))
    np.testing.assert_allclose(sampler.std_devs, np.std(sampler.data_scaled, axis=0))

    sampler.rd()
    # rd should create a 3‑component matrix (or fewer if n_d < 3)
    expected_k = min(3, sampler.n_d)
    assert sampler.data_reduced.shape == (sampler.n_s, expected_k)


def test_high_dimensional_path(high_dim_data, var_limits):
    """When n_d > 3 the internal Reducer should be used."""
    sampler = ActiveSampling(
        data=high_dim_data,
        n_r=3,
        vlim=var_limits,
        kernel_type=["Gaussian"],
        bw_method=TUNING_METHOD.SCOTT.name,
        h=[0.1] * 3,
    )
    # The reducer is instantiated only for n_d > 3
    assert sampler.reducer is not None
    # After rd the reduced data should have shape (n_s, n_r)
    sampler.rd()
    assert sampler.data_reduced.shape == (high_dim_data.shape[0], sampler.n_r)


# ----------------------------------------------------------------------
# Projection test
# ----------------------------------------------------------------------
def test_project_rd_to_original_space(high_dim_data, var_limits):
    sampler = ActiveSampling(
        data=high_dim_data,
        n_r=3,
        vlim=var_limits,
        kernel_type=["Gaussian"],
        bw_method=TUNING_METHOD.SCOTT.name,
        h=[0.1] * 3,
    )
    sampler.rd()
    # generate a few random points in reduced space
    rnd = np.random.default_rng(42)
    reduced_samples = rnd.random((4, sampler.n_r))
    # project back – should have original dimensionality
    projected = sampler.project_rd_to_original_space(reduced_samples)
    assert projected.shape == (4, sampler.n_d)


# ----------------------------------------------------------------------
# Resampling tests
# ----------------------------------------------------------------------
def test_resample_low_dim(simple_data):
    """Resample should return an array of the same dimensionality as the input."""
    vlim = np.column_stack((np.zeros(2), np.ones(2)))
    sampler = ActiveSampling(
        data=simple_data,
        n_r=2,
        vlim=vlim,
        kernel_type=["Gaussian"],
        bw_method=TUNING_METHOD.SCOTT.name,
        h=[0.1] * 2,
    )
    # Force a small number of kernels to keep the test fast
    sampler.kernel = sampler.kernel[:1]

    samples = sampler.resample()
    # For low‑dim case the result is directly the list of samples
    assert isinstance(samples, np.ndarray)
    assert samples.shape[1] == simple_data.shape[1]  # same number of columns


def test_resample_high_dim(high_dim_data, var_limits):
    """Resample should correctly map reduced samples back to the original space."""
    sampler = ActiveSampling(
        data=high_dim_data,
        n_r=3,
        vlim=var_limits,
        kernel_type=["Gaussian"],
        bw_method=TUNING_METHOD.SCOTT.name,
        h=[0.1] * 3,
    )
    # Reduce once so that internal attributes (eigenvectors, means, etc.) exist
    sampler.rd()
    # Run resampling – this will go through the reducer branch
    samples = sampler.resample()
    assert isinstance(samples, np.ndarray)
    # Output must be in the original variable space (n_d columns)
    assert samples.shape[1] == high_dim_data.shape[1]
    # All values should respect the variable limits (within a tiny numerical tolerance)
    assert np.all(samples >= var_limits[:, 0] - 1e-8)
    assert np.all(samples <= var_limits[:, 1] + 1e-8)


# ----------------------------------------------------------------------
# Edge‑case sanity checks
# ----------------------------------------------------------------------
def test_resample_without_reduction_returns_numpy_array(simple_data):
    """When n_d <= 3 the method should return a plain NumPy array."""
    vlim = np.column_stack((np.zeros(2), np.ones(2)))
    sampler = ActiveSampling(
        data=simple_data,
        n_r=2,
        vlim=vlim,
        kernel_type=["Gaussian"],
        bw_method=TUNING_METHOD.SCOTT.name,
        h=[0.1] * 2,
    )
    out = sampler.resample()
    assert isinstance(out, np.ndarray)
    assert out.shape[1] == simple_data.shape[1]


def test_kde_resample_is_deterministic_given_seed(simple_data):
    """kde_resample uses random.choice/choices – fixing the seed should give reproducible output."""
    vlim = np.column_stack((np.zeros(2), np.ones(2)))
    sampler = ActiveSampling(
        data=simple_data,
        n_r=2,
        vlim=vlim,
        kernel_type=["Gaussian"],
        bw_method=TUNING_METHOD.SCOTT.name,
        seed=999,
        h=[0.1] * 2,
    )
    # Run twice with the same seed – the internal RNG is re‑seeded only once at init,
    # but the stochastic path is deterministic for the duration of the test.
    first = sampler.kde_resample(sampler.kernel, [1], sampler.data, seed=10000)
    second = sampler.kde_resample(sampler.kernel, [1], sampler.data, seed=10000)
    np.testing.assert_allclose(first, second)
