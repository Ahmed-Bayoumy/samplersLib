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
#  Copyright (C) 2024-2026  Ahmed H. Bayoumy                                          #
# ------------------------------------------------------------------------------------#
"""

from collections import defaultdict

import numpy as np
from scipy.stats import rankdata


class TrustWorthiness:
    def __init__(self, ref, pred, method="average"):
        self.ref = np.asarray(ref)
        self.pred = np.asarray(pred)
        self.method = method

    # Pearson correlation using NumPy
    def pearsonr(self, ref=None, pred=None):
        if ref is not None and pred is not None:
            x_mean = np.mean(ref)
            y_mean = np.mean(pred)
            numerator = np.sum((ref - x_mean) * (pred - y_mean))
            denominator = np.sqrt(np.sum((ref - x_mean) ** 2) * np.sum((pred - y_mean) ** 2))
        else:
            x_mean = np.mean(self.ref)
            y_mean = np.mean(self.pred)
            numerator = np.sum((self.ref - x_mean) * (self.pred - y_mean))
            denominator = np.sqrt(np.sum((self.ref - x_mean) ** 2) * np.sum((self.pred - y_mean) ** 2))

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def spearmanr(self):
        """
        Compute Spearman's rank correlation using scipy's rankdata.

        Parameters:
        - x, y: Arrays of observations
        - method: Tie-breaking method for rankdata ('average', 'min', 'max', 'dense', 'ordinal')
        """

        rank_x = rankdata(self.ref, method=self.method)
        rank_y = rankdata(self.pred, method=self.method)

        return self.pearsonr(rank_x, rank_y)

    def kendalltau_b(self):
        """
        Efficient Kendall’s tau-b using scipy's rankdata for tie handling.

        Parameters:
        - x, y: Arrays of observations
        - method: Tie method for ranking ('average', etc.)
        """

        n = len(self.ref)

        rank_x = rankdata(self.ref, method=self.method)
        rank_y = rankdata(self.pred, method=self.method)

        num_concordant = 0
        num_discordant = 0
        tie_x = 0
        tie_y = 0

        for i in range(n - 1):
            for j in range(i + 1, n):
                dx = rank_x[i] - rank_x[j]
                dy = rank_y[i] - rank_y[j]
                if dx == 0 and dy == 0:
                    continue  # both tied
                elif dx == 0:
                    tie_x += 1
                elif dy == 0:
                    tie_y += 1
                elif dx * dy > 0:
                    num_concordant += 1
                elif dx * dy < 0:
                    num_discordant += 1

        denominator = np.sqrt((num_concordant + num_discordant + tie_x) * (num_concordant + num_discordant + tie_y))

        if denominator == 0:
            return 0.0
        return (num_concordant - num_discordant) / denominator

    def kendalltau_b_fast(self):
        """
        Efficient Kendall’s tau-b using merge sort for O(n log n) and
        scipy's rankdata for tie handling.

        Parameters:
        - x, y: Arrays of observations
        - method: Tie-breaking method for rankdata ('average', 'min', etc.)
        """
        n = len(self.ref)

        # Step 1: Rank x and y using scipy's rankdata
        rank_x = rankdata(self.ref, method=self.method)
        rank_y = rankdata(self.pred, method=self.method)

        # Step 2: Sort x, reorder y accordingly
        sort_idx = np.argsort(rank_x)
        y_sorted = rank_y[sort_idx]

        # Step 3: Count discordant pairs using merge sort inversion count
        def count_inversions(arr):
            def merge_sort(arr):
                if len(arr) <= 1:
                    return arr, 0
                mid = len(arr) // 2
                left, inv_left = merge_sort(arr[:mid])
                right, inv_right = merge_sort(arr[mid:])
                merged, inv_split = merge_and_count(left, right)
                return merged, inv_left + inv_right + inv_split

            def merge_and_count(left, right):
                merged = []
                i = j = inv_count = 0
                while i < len(left) and j < len(right):
                    if left[i] <= right[j]:
                        merged.append(left[i])
                        i += 1
                    else:
                        merged.append(right[j])
                        inv_count += len(left) - i
                        j += 1
                merged.extend(left[i:])
                merged.extend(right[j:])
                return merged, inv_count

            _, count = merge_sort(list(arr))
            return count

        num_discordant = count_inversions(y_sorted)
        total_pairs = n * (n - 1) // 2
        num_concordant = total_pairs - num_discordant

        # Step 4: Count ties in x and y
        def count_ties(arr):
            tie_counts = defaultdict(int)
            for val in arr:
                tie_counts[val] += 1
            ties = sum(c * (c - 1) // 2 for c in tie_counts.values() if c > 1)
            return ties

        tie_x = count_ties(rank_x)
        tie_y = count_ties(rank_y)

        # Step 5: Compute Tau-b
        denominator = np.sqrt((total_pairs - tie_x) * (total_pairs - tie_y))
        if denominator == 0:
            return 0.0
        return (num_concordant - num_discordant) / denominator


def compute_dimension_relevance(X, y, top_k=20):
    """
    Computes Kendall tau between each dimension and the objective values.
    Higher tau → more relevant for perturbation.
    """
    top_idx = np.argsort(y)[:top_k]
    X_top = X[top_idx]
    y_top = y[top_idx]

    taus = []
    for d in range(X.shape[1]):
        tw: TrustWorthiness = TrustWorthiness(ref=X_top[:, d], pred=y_top)
        tau = tw.kendalltau_b_fast()
        taus.append(abs(tau))  # Take abs because direction doesn’t matter

    taus = np.nan_to_num(taus)  # Replace nan with 0
    taus = np.array(taus)
    taus /= np.sum(taus) + 1e-8  # Normalize to form a probability distribution
    return taus


# x = [10, 20, 20, 40, 50]
# y = [15, 25, 25, 47, 60]

# tw = TrustWorthiness(ref=x, pred=y)
# print("Pearson r:   ", tw.pearsonr())
# print("Spearman rho:", tw.spearmanr())
# print("Kendall tau: ", tw.kendalltau_b())
