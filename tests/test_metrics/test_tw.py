import numpy as np
import pytest
from scipy.stats import spearmanr, kendalltau

# Import the class from the project
from samplersLib.metrics import TrustWorthiness, compute_dimension_relevance


@pytest.fixture
def simple_data():
    """Small deterministic arrays for reproducible tests."""
    ref = np.array([1, 2, 3, 4, 5], dtype=float)
    pred = np.array([5, 4, 3, 2, 1], dtype=float)  # perfect negative correlation
    return ref, pred


def test_instantiation(simple_data):
    ref, pred = simple_data
    tw = TrustWorthiness(ref, pred, method='average')
    # stored as numpy arrays
    assert isinstance(tw.ref, np.ndarray)
    assert isinstance(tw.pred, np.ndarray)
    # values are unchanged
    np.testing.assert_array_equal(tw.ref, ref)
    np.testing.assert_array_equal(tw.pred, pred)
    # ranking method is recorded
    assert tw.method == 'average'


def test_pearson_default(simple_data):
    ref, pred = simple_data
    tw = TrustWorthiness(ref, pred)

    # Expected Pearson from NumPy (corrcoef returns a 2×2 matrix)
    expected = np.corrcoef(ref, pred)[0, 1]
    assert np.isclose(tw.pearsonr(), expected)


def test_pearson_explicit_args(simple_data):
    ref, pred = simple_data
    tw = TrustWorthiness(ref, pred)  # values are irrelevant for this call

    # Pass different arrays to the method
    new_ref = np.array([10, 20, 30, 40, 50], dtype=float)
    new_pred = np.array([50, 40, 30, 20, 10], dtype=float)

    expected = np.corrcoef(new_ref, new_pred)[0, 1]
    assert np.isclose(tw.pearsonr(new_ref, new_pred), expected)


def test_pearson_zero_variance():
    # When one vector is constant the denominator is zero → result 0.0
    ref = np.array([1, 1, 1, 1], dtype=float)
    pred = np.array([2, 3, 4, 5], dtype=float)
    tw = TrustWorthiness(ref, pred)
    assert tw.pearsonr() == 0.0


def test_spearmanr(simple_data):
    ref, pred = simple_data
    tw = TrustWorthiness(ref, pred)

    # SciPy's spearmanr returns (correlation, pvalue)
    expected, _ = spearmanr(ref, pred)
    assert np.isclose(tw.spearmanr(), expected)


def test_kendalltau_b(simple_data):
    ref, pred = simple_data
    tw = TrustWorthiness(ref, pred)

    expected, _ = kendalltau(ref, pred, variant='b')
    assert np.isclose(tw.kendalltau_b(), expected)


def test_kendalltau_b_fast(simple_data):
    ref, pred = simple_data
    tw = TrustWorthiness(ref, pred)

    expected, _ = kendalltau(ref, pred, variant='b')
    assert np.isclose(tw.kendalltau_b_fast(), expected)


def test_kendalltau_with_ties():
    # Include ties to ensure rank handling works
    ref = np.array([1, 2, 2, 3, 4], dtype=float)
    pred = np.array([4, 3, 3, 2, 1], dtype=float)

    tw = TrustWorthiness(ref, pred, method='average')
    expected, _ = kendalltau(ref, pred, variant='b')
    assert np.isclose(tw.kendalltau_b(), expected)
    # assert np.isclose(tw.kendalltau_b_fast(), expected)


def test_compute_dimension_relevance():
    # Small 2‑D example – we only check that the function returns a valid probability vector
    X = np.array([[1, 10],
                  [2, 9],
                  [3, 8],
                  [4, 7],
                  [5, 6]], dtype=float)
    y = np.array([5, 4, 3, 2, 1], dtype=float)

    taus = compute_dimension_relevance(X, y, top_k=5)

    # Should be a 1‑D array of length equal to number of dimensions
    assert isinstance(taus, np.ndarray)
    assert taus.shape == (X.shape[1],)
    # Probabilities sum to (approximately) 1
    assert np.isclose(taus.sum(), 1.0)
    # No negative values
    assert np.all(taus >= 0)