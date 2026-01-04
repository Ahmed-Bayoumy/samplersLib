# tests/test_decision_tree.py
import numpy as np
import pytest
from collections import Counter

# Import the class from the module you posted
from samplersLib.importance import DecisionTree, gini_impurity, calculate_info_gain


# ----------------------------------------------------------------------
# Helper data sets
# ----------------------------------------------------------------------
@pytest.fixture
def simple_binary_data():
    """Two‑feature, binary‑class toy data set."""
    X = np.array([[0, 1],
                  [0, 2],
                  [1, 1],
                  [1, 2]])
    y = np.array([0, 0, 1, 1])
    return X, y


@pytest.fixture
def multi_class_data():
    """Three‑class data set with a clear split on the first feature."""
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1],
                  [2, 0],
                  [2, 1]])
    y = np.array([0, 0, 1, 1, 2, 2])
    return X, y


# ----------------------------------------------------------------------
# Basic sanity checks
# ----------------------------------------------------------------------
def test_gini_impurity_binary(simple_binary_data):
    _, y = simple_binary_data
    # Two of each class → impurity = 0.5
    assert pytest.approx(gini_impurity(y), 0.001) == 0.5


def test_info_gain_simple_split(simple_binary_data):
    X, y = simple_binary_data
    # Splitting on feature 0 at threshold 0.5 separates the classes perfectly
    gain = calculate_info_gain(X, y, feature_index=0, threshold=0.5)
    # Perfect split → impurity goes from 0.5 to 0 → gain = 0.5
    assert pytest.approx(gain, 0.001) == 0.5


# ----------------------------------------------------------------------
# DecisionTree instantiation & fitting
# ----------------------------------------------------------------------
def test_tree_instantiation():
    tree = DecisionTree(max_depth=2)
    assert isinstance(tree, DecisionTree)
    assert tree.max_depth == 2
    assert tree.tree is None


def test_tree_fit_binary(simple_binary_data):
    X, y = simple_binary_data
    tree = DecisionTree(max_depth=1)
    tree.fit(X, y)

    # With depth 1 the tree should consist of a single split dict
    assert isinstance(tree.tree, dict)
    assert set(tree.tree.keys()) == {"feature_index", "threshold", "left", "right"}

    # The chosen split should be on feature 0 (the only perfect split)
    assert tree.tree["feature_index"] == 0
    # Threshold can be any value between the two distinct values (0 and 1)
    assert 0 <= tree.tree["threshold"] < 1


def test_tree_predict_binary(simple_binary_data):
    X, y = simple_binary_data
    tree = DecisionTree(max_depth=1)
    tree.fit(X, y)

    preds = tree.predict(X)
    # Because the split perfectly separates the classes, predictions must match y
    assert np.array_equal(preds, y)


def test_tree_fit_multi_class(multi_class_data):
    X, y = multi_class_data
    tree = DecisionTree(max_depth=2)
    tree.fit(X, y)

    # The root split should be on feature 0 (values 0,1,2) with a threshold that
    # separates class 0 from the others (e.g., 0.5)
    assert isinstance(tree.tree, dict)
    assert tree.tree["feature_index"] == 0
    assert 0 <= tree.tree["threshold"] < 1.5  # any threshold between 0 and 1 works


def test_tree_predict_multi_class(multi_class_data):
    X, y = multi_class_data
    tree = DecisionTree(max_depth=2)
    tree.fit(X, y)

    preds = tree.predict(X)
    # With depth 2 the tree can perfectly separate the three classes in this toy set
    assert np.array_equal(preds, y)


# ----------------------------------------------------------------------
# Feature importance
# ----------------------------------------------------------------------
def test_feature_importance_binary(simple_binary_data):
    X, y = simple_binary_data
    tree = DecisionTree(max_depth=1)
    tree.fit(X, y)

    importance = tree.get_feature_importance(X, y)
    # Only feature 0 is used for the split, so its importance should be > 0
    assert importance[0] > 0
    # Feature 1 is never used → importance should be 0
    assert importance[1] == 0


def test_feature_importance_multi_class(multi_class_data):
    X, y = multi_class_data
    tree = DecisionTree(max_depth=2)
    tree.fit(X, y)

    importance = tree.get_feature_importance(X, y)
    # Feature 0 is the primary splitter; feature 1 may get a small contribution
    assert importance[0] > importance[1]
    assert importance.sum() > 0

# test_tree_fit_binary(simple_binary_data())