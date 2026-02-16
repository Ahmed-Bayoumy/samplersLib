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

import numpy as np


class Particle:
    def __init__(self, bounds, pos: np.ndarray = None, seed: int = 12345):
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        if pos is None:
            self.position = self.rng.uniform(bounds[0], bounds[1], size=(bounds.shape[1],))
        else:
            self.position = pos.copy()
        self.velocity = self.rng.uniform(-1, 1, size=(bounds.shape[1],))
        self.best_position = self.position.copy()
        self.best_value = float("-inf")  # For sampling, we want the highest likelihood

    def update_velocity(self, global_best_position, inertia_weight, cognitive_weight, social_weight):
        r1, r2 = self.rng.random(2)
        self.velocity = (
            inertia_weight * self.velocity
            + cognitive_weight * r1 * (self.best_position - self.position)
            + social_weight * r2 * (global_best_position - self.position)
        )

    def update_position(self, bounds):
        self.position += self.velocity
        # Enforce bounds
        self.position = np.clip(self.position, bounds[0], bounds[1])

    def evaluate(self, target_distribution):
        value = target_distribution(self.position)
        if value > self.best_value:
            self.best_value = value
            self.best_position = self.position.copy()
