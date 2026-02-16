import numpy as np
import pytest

# Import the class under test
from samplersLib.samplers import TUNING_METHOD, BiTPE, Gaussian


# Helper to create tiny synthetic data
# pylint: disable=missing-function-docstring
def make_data(num_points=5, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.uniform(0, 1, size=(num_points, dim))
    f_vals = rng.normal(loc=0.0, scale=1.0, size=num_points)
    return data, f_vals


@pytest.fixture
# pylint: disable=missing-function-docstring
def bitpe_instance():
    # Small synthetic “good” and “bad” datasets
    good_data, good_f = make_data(num_points=4, dim=3, seed=1)
    bad_data, bad_f = make_data(num_points=3, dim=3, seed=2)

    # Use a deterministic seed so the test is reproducible
    v = np.array([[0, 1]] * 3)
    return BiTPE(
        good_data=good_data,
        good_f_values=good_f,
        bad_data=bad_data,
        bad_f_values=bad_f,
        kernel_type={"Gaussian": 1},
        n_r=2,  # keep the runtime tiny
        vlim=v,
        bw_method=TUNING_METHOD.SCOTT.name,
        seed=12345,
        weights=None,
        h=np.array([0.1]).tolist(),
        gamma=0.25,
    )


# pylint: disable=missing-function-docstring
def test_instantiation_and_kernel_creation(bitpe_instance):
    # The instance should have sorted observations
    assert bitpe_instance.good_obs.shape == bitpe_instance.good_data.shape
    assert bitpe_instance.bad_obs.shape == bitpe_instance.bad_data.shape

    # Kernels list must contain one kernel per dimension
    dim = len(bitpe_instance.kernels.keys())
    assert len(bitpe_instance.good_kernel) == dim
    assert len(bitpe_instance.bad_kernel) == dim

    # All kernels should be instances of the expected class (Gaussian here)
    for k in bitpe_instance.good_kernel + bitpe_instance.bad_kernel:
        assert isinstance(k, Gaussian)


# pylint: disable=missing-function-docstring
def test_rank_data_and_initialize_kernels(bitpe_instance):
    # The good observations must be sorted by the associated f‑values
    sorted_idx = np.argsort(bitpe_instance.good_f_values)
    np.testing.assert_array_equal(bitpe_instance.good_obs, bitpe_instance.good_data[sorted_idx])

    # Same check for the bad observations
    sorted_idx_bad = np.argsort(bitpe_instance.bad_f_values)
    np.testing.assert_array_equal(bitpe_instance.bad_obs, bitpe_instance.bad_data[sorted_idx_bad])


# pylint: disable=missing-function-docstring
def test_suggest_returns_candidates(bitpe_instance):
    # Run the suggestion routine (uses the small n_r defined in the fixture)
    candidates = bitpe_instance._suggest()

    # Should always return a NumPy array
    assert isinstance(candidates, np.ndarray)

    # With n_r=2 we expect at most 2 candidates; could be 0 if no ratio > 1
    assert candidates.shape[0] <= 2

    # If any candidate is produced, its dimensionality must match the data
    if candidates.shape[0] > 0:
        assert candidates.shape[1] == bitpe_instance.good_data.shape[1]

    # Basic sanity check: the method should not raise and should produce finite numbers
    assert np.all(np.isfinite(candidates)) if candidates.size else True


# pylint: disable=missing-function-docstring
def test_combined_kernels_rosen(plotting=False):
    v = np.array([[-5.0, 10.0]] * 10)

    data = [
        [-2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5],
        [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -2.5],
        [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, -1.5],
        [0.75, 0.75, 0.75, 0.5, 0.0, -0.5, 0.0, 0.5, 0.25, -1.25],
        [0.8125, 1.0, 1.0625, 0.75, 0.0625, -0.1875, 0.3125, 0.5625, 0.25, -0.5],
    ]

    data_f = [69016.5, 1226.5, 462.5, 247.6875, 119.9593505859375]

    bad_points = [
        [-2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5],
        [-2.5, -2.5, -1.5, -2.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        [-2.5, -1.5, -1.5, -2.5, -1.5, -1.5, -1.5, -2.5, -1.5, -1.5],
        [-1.5, -1.5, -1.5, -2.5, -1.5, -2.5, -1.5, -1.5, -1.5, -2.5],
        [-1.5, 0.5, -1.5, -0.5, 0.5, -2.5, -1.5, 0.5, 0.5, -0.5],
        [-1.5, 0.5, -1.5, -0.5, 0.5, -2.5, 0.5, -1.5, 0.5, -0.5],
        [-1.5, 0.5, -1.5, -0.5, 0.5, -0.5, -1.5, -1.5, 0.5, -0.5],
        [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -2.5],
        [-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -1.5],
        [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, -1.5],
        [0.25, 0.25, 0.25, 0.5, -0.25, -0.5, 0.75, 0.5, 0.0, -1.75],
        [0.5, 0.5, 0.5, 0.75, -0.25, -0.25, 0.75, 0.75, 0.75, -1.0],
        [0.75, 0.75, 0.75, 0.5, -0.5, 0.0, 0.5, 0.5, 0.5, -1.25],
        [0.75, 0.75, 0.75, 0.5, 0.0, -0.5, 0.0, 0.5, 0.25, -1.25],
        [0.75, 0.8125, 0.75, 0.5625, 0.25, -0.5, 0.0, 0.5, 0.3125, -1.1875],
        [0.75, 0.8125, 0.75, 0.6875, 0.0, -0.25, -0.0625, 0.3125, 0.375, -1.1875],
        [0.8125, 0.6875, 0.75, 0.25, 0.0, -0.3125, 0.125, 0.375, 0.0, -1.3125],
        [0.8125, 0.75, 0.8125, 0.5, 0.0625, -0.4375, 0.0625, 0.5625, 0.25, -1.0],
        [0.8125, 1.0, 1.0625, 0.75, 0.0625, -0.1875, 0.3125, 0.5625, 0.25, -0.5],
    ]

    bad_f = [
        69016.5,
        29030.5,
        28230.5,
        24474.5,
        8540.5,
        6140.5,
        3536.5,
        1226.5,
        620.5,
        462.5,
        427.765625,
        415.84375,
        336.859375,
        247.6875,
        240.82879638671875,
        237.191650390625,
        220.55828857421875,
        183.7786865234375,
        119.9593505859375,
    ]

    f_values = [rosen(x) for x in data]
    best_fx = min(f_values)

    n_new_samples = 100

    kde = BiTPE(
        good_data=data,
        good_f_values=data_f,
        bad_data=bad_points,
        bad_f_values=bad_f,
        n_r=n_new_samples,
        vlim=v,
        kernel_type={"Cauchy": 0.5, "Gaussian": 0.5},
        h=[0.0001] * 10,
        seed=12345,
    )
    candidates = kde._suggest()
    print(candidates)
    print("Initial best f(x):", best_fx)

    results = [rosen(c) for c in candidates]
    print("Final best f(x):", min(results))

    assert 1 - (min(results) / min(data_f)) > 0.85


# pylint: disable=missing-function-docstring
def rosen(x):
    x = np.array(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
