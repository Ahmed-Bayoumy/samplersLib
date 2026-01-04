
# pylint: disable=too-many-lines
"""
# ------------------------------------------------------------------------------------#
#  Samplers Library - (SamplersLib)                                                   #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU Lesser General Public License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.   #
#                                                                                     #
#  You should have received a copy of the GNU Lesser General Public License along     #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on SamplersLib at                                         #
#  https://github.com/Ahmed-Bayoumy/samplersLib                                       #
#  Copyright (C) 2022  Ahmed H. Bayoumy                                               #
# ------------------------------------------------------------------------------------#
"""

import numpy as np
from ._common import *
from collections import Counter

# Step 1: Define Gini Impurity and splitting functions

# Gini Impurity for classification
def gini_impurity(y):
  class_counts = Counter(y)
  total_samples = len(y)
  impurity = 1 - sum((count / total_samples) ** 2 for count in class_counts.values())
  return impurity

# Function to split data by a feature and threshold
def split_data(X, y, feature_index, threshold):
  left_mask = X[:, feature_index] <= threshold
  right_mask = ~left_mask
  return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

# Information Gain calculation: Reduction in impurity after split
def calculate_info_gain(X, y, feature_index, threshold):
  left_X, right_X, left_y, right_y = split_data(X, y, feature_index, threshold)
  
  # Compute Gini impurity for left and right splits
  left_impurity = gini_impurity(left_y)
  right_impurity = gini_impurity(right_y)
  
  # Weighted impurity after the split
  total_samples = len(y)
  left_weight = len(left_y) / total_samples
  right_weight = len(right_y) / total_samples
  weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
  
  # Information Gain is the reduction in impurity
  return gini_impurity(y) - weighted_impurity


# Step 2: Define a Decision Tree Class

class DecisionTree:
  def __init__(self, max_depth=None):
    self.max_depth = max_depth
    self.tree = None
  
  def fit(self, X, y):
    self.tree = self._build_tree(X, y, depth=0)
  
  def _build_tree(self, X, y, depth):
    num_samples, num_features = X.shape
    num_classes = len(np.unique(y))
    
    # Stopping conditions
    if num_classes == 1 or num_samples < 2 or (self.max_depth and depth == self.max_depth):
      return Counter(y).most_common(1)[0][0]
    
    best_split = None
    best_info_gain = -float("inf")
    best_left_X = best_right_X = best_left_y = best_right_y = None
    
    for feature_index in range(num_features):
      # Try all unique values of the feature to find the best threshold
      thresholds = np.unique(X[:, feature_index])
      for threshold in thresholds:
        info_gain = calculate_info_gain(X, y, feature_index, threshold)
        if info_gain > best_info_gain:
          best_info_gain = info_gain
          best_split = (feature_index, threshold)
          left_X, right_X, left_y, right_y = split_data(X, y, feature_index, threshold)
    
    if best_split:
      feature_index, threshold = best_split
      left_tree = self._build_tree(left_X, left_y, depth + 1)
      right_tree = self._build_tree(right_X, right_y, depth + 1)
      return {'feature_index': feature_index, 'threshold': threshold, 
              'left': left_tree, 'right': right_tree}
    else:
      return Counter(y).most_common(1)[0][0]
  
  def predict(self, X):
    return [self._predict_one(x, self.tree) for x in X]
  
  def _predict_one(self, x, tree):
    if isinstance(tree, dict):
      if x[tree['feature_index']] <= tree['threshold']:
        return self._predict_one(x, tree['left'])
      else:
        return self._predict_one(x, tree['right'])
    else:
      return tree
  
  def get_feature_importance(self, X, y):
    feature_importance = np.zeros(X.shape[1])
    self._calculate_feature_importance(X, y, self.tree, feature_importance)
    return feature_importance
  
  def _calculate_feature_importance(self, X, y, tree, feature_importance):
    if isinstance(tree, dict):
      feature_index = tree['feature_index']
      threshold = tree['threshold']
      left_X, right_X, left_y, right_y = split_data(X, y, feature_index, threshold)
      
      # Calculate the impurity reduction at this node
      info_gain = calculate_info_gain(X, y, feature_index, threshold)
      feature_importance[feature_index] += info_gain
      
      # Recurse into the left and right subtrees
      self._calculate_feature_importance(left_X, left_y, tree['left'], feature_importance)
      self._calculate_feature_importance(right_X, right_y, tree['right'], feature_importance)

class RandomForest:
  def __init__(self, n_estimators=100, max_depth=None, seed=12345):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.trees = []
    self.seed = seed
  
  def fit(self, X, y):
    self.trees = [self._train_tree(X, y) for _ in range(self.n_estimators)]
  
  def _train_tree(self, X, y):
    # Bootstrap sampling
    np.random.seed(self.seed)
    indices = np.random.choice(len(X), len(X), replace=True)
    X_bootstrap = X[indices]
    y_bootstrap = y[indices]
    
    tree = DecisionTree(max_depth=self.max_depth)
    tree.fit(X_bootstrap, y_bootstrap)
    return tree
  
  def predict(self, X):
    predictions = np.array([tree.predict(X) for tree in self.trees])
    return [Counter(pred).most_common(1)[0][0] for pred in predictions.T]
  
  def get_feature_importance(self, X, y):
    total_importance = np.zeros(X.shape[1])
    for tree in self.trees:
      total_importance += tree.get_feature_importance(X, y)
    return total_importance / self.n_estimators