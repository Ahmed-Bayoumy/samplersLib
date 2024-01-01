from enum import auto, Enum
from dataclasses import dataclass
from typing import List
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
class TUNING_METHOD(Enum):
  MLCV: int = auto()
  SCOTT: int = auto()
  SILVERMAN: int = auto()

class SAMPLING_METHOD(Enum):
  FULLFACTORIAL: int = auto()
  LH: int = auto()
  RS: int = auto()
  HALTON: int = auto()

@dataclass
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
