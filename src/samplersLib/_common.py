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

from enum import auto, Enum
from dataclasses import dataclass
from typing import List
import numpy as np
import plotly.express as px
# import seaborn as sns

# pylint: disable=missing-function-docstring
class TUNING_METHOD(Enum):
  MLCV: int = auto()
  SCOTT: int = auto()
  SILVERMAN: int = auto()

# pylint: disable=missing-function-docstring
class SAMPLING_METHOD(Enum):
  FULLFACTORIAL: int = auto()
  LH: int = auto()
  RS: int = auto()
  HALTON: int = auto()

# pylint: disable=missing-function-docstring
class KERNEL_TYPE(Enum):
  PARAMETRIC: int = auto()
  NONPARAMETRIC: int = auto()

@dataclass
# pylint: disable=missing-function-docstring
class eq_solvers:
  a: np.ndarray = None
  b: np.ndarray = None

  def __init__(self, a: np.ndarray, b:np.ndarray):
    self.a =np.atleast_2d(a)
    self.b = np.atleast_1d(b)

  def fwd_solve(self):
    n = len(self.b)
    x = [0]*n
    for i in range(n):
      x[i] = self.b[i]
      for j in range(0,i):
        x[i] -= self.a[i][j]*x[j]
        x[i]/= self.a[i][i]

    return x
