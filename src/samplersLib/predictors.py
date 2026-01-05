# pylint: disable=too-many-lines
"""
# ------------------------------------------------------------------------------------#
#  Samplers Library - (SamplersLib)                                                   #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the BSD 3-Clause License as published by the Free Software                #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the BSD 3-Clause License for more details.                #
#                                                                                     #
#  You should have received a copy of the BSD 3-Clause License along                  #
#  with this program. If not, see <https://opensource.org/license/bsd-3-clause/>.     #
#                                                                                     #
#  You can find information on SamplersLib at                                         #
#  https://github.com/Ahmed-Bayoumy/samplersLib                                       #
#  Copyright (C) 2026  Ahmed H. Bayoumy                                               #
# ------------------------------------------------------------------------------------#
"""

import heapq
from typing import Callable
from scipy.stats import kendalltau, norm
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod

import numpy as np
# ====================== PREDICTOR CLASSES ======================

class Predictor(ABC):
    """_summary_

    :param ABC: _description_
    :type ABC: _type_
    :return: _description_
    :rtype: _type_
    """
    @abstractmethod
    def fit(self, X, y):
        """_summary_

        :param X: _description_
        :type X: _type_
        :param y: _description_
        :type y: _type_
        """
        pass

    @abstractmethod
    def predict(self, x):
        """_summary_

        :param x: _description_
        :type x: _type_
        :return: _description_
        :rtype: _type_
        """
        pass

    @abstractmethod
    def uncertainty(self, x):
        """_summary_

        :param x: _description_
        :type x: _type_
        :return: _description_
        :rtype: _type_
        """
        pass


class KernelWeightedAverage(Predictor):
    """_summary_
    """
    def __init__(self, bandwidth=0.1, kw_calculator: Callable = None):
        self.mixed_kernel: Callable = kw_calculator
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x):
        weights = np.array([self.mixed_kernel(x, xi) for xi in self.X])
        return np.dot(weights, self.y) / (np.sum(weights) + 1e-8)

    def uncertainty(self, x):
        weights = np.array([self.mixed_kernel(x, xi) for xi in self.X])
        mean = self.predict(x)
        var = np.dot(weights, (self.y - mean)**2) / (np.sum(weights) + 1e-8)
        return np.sqrt(var)


class KernelRidgeRegression(Predictor):
    """_summary_
    """
    def __init__(self, alpha=1.0, bandwidth=0.1, kw_calculator: Callable = None):
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.mixed_kernel: Callable = kw_calculator

    def fit(self, X, y):
        self.X = X
        self.y = y
        K = np.array([[self.mixed_kernel(xi, xj) for xj in X] for xi in X])
        self.alpha_vec = np.linalg.solve(K + self.alpha * np.eye(len(X)), y)

    def predict(self, x):
        k = np.array([self.mixed_kernel(x, xi) for xi in self.X])
        return np.dot(k, self.alpha_vec)

    def uncertainty(self, x):
        return 0.1  # approximate or fixed for simplicity


class LocalPolynomialRegression(Predictor):
    """_summary_
    """
    def __init__(self, bandwidth=0.2, kw_calculator: Callable = None):
        self.bandwidth = bandwidth
        self.mixed_kernel: Callable = kw_calculator

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x):
        W = np.diag([self.mixed_kernel(x, xi) for xi in self.X])
        xc = self.X - x
        A = np.hstack([np.ones((len(self.X), 1)), xc])
        beta = np.linalg.pinv(A.T @ W @ A) @ A.T @ W @ self.y
        return beta[0]

    def uncertainty(self, x):
        return 0.1  # same, or compute from residuals


class GPWithMixedKernel(Predictor):
    """_summary_
    """
    def __init__(self, noise=1e-5, bandwidth=0.1, kw_calculator: Callable = None):
        self.mixed_kernel: Callable = kw_calculator
        self.noise = noise
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.K = np.array([[self.mixed_kernel(xi, xj) for xj in X] for xi in X])
        self.K += self.noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(self.K)

    def predict(self, x):
        k = np.array([self.mixed_kernel(x, xi) for xi in self.X])
        return np.dot(k, np.dot(self.K_inv, self.y))

    def uncertainty(self, x):
        k = np.array([self.mixed_kernel(x, xi) for xi in self.X])
        kx = self.mixed_kernel(x, x)
        return np.sqrt(kx - np.dot(k, np.dot(self.K_inv, k.T)))


class MDNInspired(Predictor):
    """_summary_
    """
    def __init__(self, bandwidth=0.2, kw_calculator: Callable = None):
        self.mixed_kernel: Callable = kw_calculator
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x):
        weights = np.array([self.mixed_kernel(x, xi) for xi in self.X])
        weights = weights / (np.sum(weights) + 1e-8)
        return np.sum(weights * self.y)

    def uncertainty(self, x):
        weights = np.array([self.mixed_kernel(x, xi) for xi in self.X])
        weights /= (np.sum(weights) + 1e-8)
        mean = self.predict(x)
        return np.sqrt(np.sum(weights * (self.y - mean)**2))


class KDNode(Predictor):
    """_summary_
    """
    def __init__(self, point, label, axis, index=None, left=None, right=None):
        self.point = point
        self.label = label
        self.axis = axis
        self.index = index
        self.left = left
        self.right = right

    def fit(self, X, y):
        pass

    def predict(self, x):
        pass

    def uncertainty(self, x):
        pass


class KDTree(Predictor):
    """_summary_

    :param Predictor: _description_
    :type Predictor: _type_
    """
    def __init__(self, X, y):
        self.root = self.build_tree(X, y)

    def fit(self, X, y):
        pass

    def predict(self, x):
        pass

    def uncertainty(self, x):
        pass

    def build_tree(self, X, y, indices=None, depth=0):
        """_summary_

        :param X: _description_
        :type X: _type_
        :param y: _description_
        :type y: _type_
        :param indices: _description_, defaults to None
        :type indices: _type_, optional
        :param depth: _description_, defaults to 0
        :type depth: int, optional
        :return: _description_
        :rtype: _type_
        """
        if indices is None:
            indices = np.arange(len(X))

        if len(indices) == 0:
            return None

        k = X.shape[1]
        axis = depth % k
        sorted_indices = indices[np.argsort(X[indices, axis])]
        median = len(sorted_indices) // 2
        median_idx = sorted_indices[median]

        return KDNode(
            point=X[median_idx],
            label=y[median_idx],
            axis=axis,
            index=median_idx,
            left=self.build_tree(X, y, sorted_indices[:median], depth + 1),
            right=self.build_tree(X, y, sorted_indices[median + 1:], depth + 1)
        )

    def _knn(self, node, target, k, heap):
        """_summary_

        :param node: _description_
        :type node: _type_
        :param target: _description_
        :type target: _type_
        :param k: _description_
        :type k: _type_
        :param heap: _description_
        :type heap: _type_
        """
        if node is None:
            return

        dist = np.linalg.norm(node.point - target)
        heapq.heappush(heap, (-dist, node.point.tolist(), node.label))
        if len(heap) > k:
            heapq.heappop(heap)

        axis = node.axis
        diff = target[axis] - node.point[axis]

        close, away = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._knn(close, target, k, heap)

        # Check if we need to explore the other branch
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(away, target, k, heap)

    def query(self, x, k=1):
        """_summary_

        :param x: _description_
        :type x: _type_
        :param k: _description_, defaults to 1
        :type k: int, optional
        :return: _description_
        :rtype: _type_
        """
        heap = []
        self._knn(self.root, np.array(x), k, heap)

        # Return (distance, index, label)
        return sorted([(-d, idx, l) for d, idx, l in heap], key=lambda t: t[0])


class KNNKernelWeighted(Predictor):
    """_summary_

    :param Predictor: _description_
    :type Predictor: _type_
    """
    def __init__(self, k=10, bandwidth=0., kw_calculator: Callable = None):
        self.mixed_kernel: Callable = kw_calculator
        self.k = k
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.kdtree = KDTree(self.X, self.y)  # Build once

    def _get_k_nearest(self, x):
        results = self.kdtree.query(x, self.k)

        distances = np.array([d for d, _, _ in results])
        indices = np.array([
            np.where((self.X == p).all(axis=1))[0][0]
            for _, p, _ in results
        ])

        return distances, indices

    def predict(self, x):
        x = np.array(x)
        distances, indices = self._get_k_nearest(x)
        neighbors = self.X[indices]
        values = self.y[indices]

        weights = np.array([self.mixed_kernel(x, xi) for xi in neighbors])
        weights /= (np.sum(weights) + 1e-8)

        return np.dot(weights, values)

    def uncertainty(self, x):
        x = np.array(x)
        distances, indices = self._get_k_nearest(x)
        neighbors = self.X[indices]
        values = self.y[indices]

        mean = self.predict(x)

        weights = np.array([self.mixed_kernel(x, xi) for xi in neighbors])
        weights /= (np.sum(weights) + 1e-8)

        return np.sqrt(np.sum(weights * (values - mean) ** 2))

# --- Adaptive Ensemble Model ---


class AdaptiveEnsemble(Predictor):
    """_summary_

    :param Predictor: _description_
    :type Predictor: _type_
    """
    def __init__(self, models, bandwidth=0.1):
        self.models = models
        self.n_models = len(models)
        self.weights = np.ones(self.n_models) / self.n_models  # uniform start
        self.bandwidth = bandwidth
        self.history_predictions = []  # Store recent predictions for weight updates
        self.history_targets = []

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)

    def predict(self, x):
        preds = np.array([m.predict(x) for m in self.models])
        # Weighted average prediction
        pred_ensemble = np.dot(self.weights, preds)
        return pred_ensemble

    def uncertainty(self, x):
        # Combine uncertainties weighted plus variance of predictions
        preds = np.array([m.predict(x) for m in self.models])
        uncerts = np.array([m.uncertainty(x) for m in self.models])
        mean_pred = np.dot(self.weights, preds)
        variance = np.dot(self.weights, (preds - mean_pred)**2)
        weighted_uncert = np.dot(self.weights, uncerts)
        # Combine uncertainty + variance of predictions
        return np.sqrt(weighted_uncert**2 + variance)


    def predict_with_zscore(self, xps, y_obs=None):
        """_summary_

        :param xps: _description_
        :type xps: _type_
        :param y_obs: _description_, defaults to None
        :type y_obs: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        # 1. Get predictions and individual model uncertainties
        z_score = []
        mean_pred = []
        total_uncertainty = []
        for i, x in enumerate(xps):
            preds = np.array([m.predict(x) for m in self.models])
            uncerts = np.array([m.uncertainty(x) for m in self.models])

            # 2. Calculate ensemble mean (the mu for z-score)
            mean_pred.append(np.dot(self.weights, preds))

            # 3. Calculate combined uncertainty (the sigma for z-score)
            # Variance of predictions (Epistemic) + Mean of individual uncertainties (Aleatoric)
            variance_of_preds = np.dot(self.weights, (preds - mean_pred[i])**2)
            weighted_uncert_sq = np.dot(self.weights, uncerts**2)

            total_uncertainty.append(np.sqrt(weighted_uncert_sq + variance_of_preds))

            # 4. Calculate Z-score if an observed value is provided
            # Formula: z = (observed - mean) / total_standard_deviation
            if y_obs is not None and y_obs[i] is not None:
                z_score.append((y_obs[i] - mean_pred[i]) / total_uncertainty[i])

        return z_score

    def calculate_testing_error(self, xps, yps):
        """_summary_

        :param xps: _description_
        :type xps: _type_
        :param yps: _description_
        :type yps: _type_
        :return: _description_
        :rtype: _type_
        """
        mean_MSE = 0.0
        for i, x in enumerate(xps):
            preds = np.array([m.predict(x) for m in self.models])
            mean_MSE += (yps[i] - np.dot(self.weights, preds))**2
        mean_MSE /= i
        return mean_MSE

    def update_weights(self, X_val, y_val):
        """_summary_

        :param X_val: _description_
        :type X_val: _type_
        :param y_val: _description_
        :type y_val: _type_
        """
        # Evaluate all models on validation points X_val with true y_val
        preds = np.array([[m.predict(x) for x in X_val] \
                          for m in self.models])  # shape: (n_models, n_points)
        errors = np.abs(preds - y_val)
        mean_errors = errors.mean(axis=1) + 1e-8  # avoid div by zero

        # Convert errors to weights: smaller error → higher weight
        inv_errors = 1 / mean_errors
        new_weights = inv_errors / np.sum(inv_errors)

        # Smooth update with momentum (optional)
        alpha = 0.7
        self.weights = alpha * self.weights + (1 - alpha) * new_weights
        self.weights /= np.sum(self.weights)  # normalize
