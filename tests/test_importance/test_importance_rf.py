import numpy as np
import pytest
from collections import Counter
from samplersLib.importance import RandomForest

# ----------------------------------------------------------------------
# Helper fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def simple_data():
    """A tiny deterministic dataset for quick testing."""
    # 4 samples, 2 features
    X = np.array([[0, 1],
                  [1, 1],
                  [0, 0],
                  [1, 0]])
    # Binary target – easy to separate
    y = np.array([0, 0, 1, 1])
    return X, y


# ----------------------------------------------------------------------
# Mock DecisionTree (so we don’t depend on the real implementation)
# ----------------------------------------------------------------------
class DummyTree:
    """A very small stand‑in for DecisionTree that records calls."""
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.fitted = False
        self.last_X = None
        self.last_y = None

    def fit(self, X, y):
        self.fitted = True
        self.last_X = X
        self.last_y = y

    def predict(self, X):
        # Return the majority class of the training set for every row
        majority = Counter(self.last_y).most_common(1)[0][0]
        return np.full(X.shape[0], majority)

    def get_feature_importance(self, X, y):
        # Return a dummy importance vector (all ones)
        return np.ones(X.shape[1])


# Patch the real DecisionTree with our DummyTree for the duration of the tests
@pytest.fixture(autouse=True)
def patch_decision_tree(monkeypatch):
    monkeypatch.setattr('samplersLib.importance.DecisionTree', DummyTree)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_instantiation_defaults():
    rf = RandomForest()
    assert rf.n_estimators == 100
    assert rf.max_depth is None
    assert rf.trees == []


def test_instantiation_custom_params():
    rf = RandomForest(n_estimators=5, max_depth=3)
    assert rf.n_estimators == 5
    assert rf.max_depth == 3
    assert rf.trees == []


def test_fit_creates_correct_number_of_trees(simple_data):
    X, y = simple_data
    rf = RandomForest(n_estimators=7, max_depth=2)
    rf.fit(X, y)

    # Should have exactly 7 DummyTree instances
    assert len(rf.trees) == 7
    for tree in rf.trees:
        assert isinstance(tree, DummyTree)
        assert tree.fitted is True
        # Verify that each tree received a bootstrap sample of the right shape
        assert tree.last_X.shape == X.shape
        assert tree.last_y.shape == y.shape


def test_predict_majority_vote(simple_data):
    X, y = simple_data
    rf = RandomForest(n_estimators=3)
    rf.fit(X, y)

    # All DummyTree.predict return the majority class of the training set (0)
    preds = rf.predict(X)
    # With three identical predictions, the majority vote must be that class
    assert preds == [0] * X.shape[0]


def test_feature_importance_averaging(simple_data):
    X, y = simple_data
    rf = RandomForest(n_estimators=4)
    rf.fit(X, y)

    # Each DummyTree returns a vector of ones → sum = n_estimators
    importance = rf.get_feature_importance(X, y)

    # After averaging we should get a vector of ones again
    expected = np.ones(X.shape[1])
    np.testing.assert_array_almost_equal(importance, expected)