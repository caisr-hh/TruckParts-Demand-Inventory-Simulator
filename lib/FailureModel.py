from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

# abstract class
class FailureModel(ABC):
    # hazard function
    @ abstractmethod
    def hazard_func(self, time: int, delta_time: int) -> float:
        ...
    
    # step probability function (conditional probability)
    @ abstractmethod
    def step_prob_func(self, time: int, delta_time: int) -> float:
        ...


# Exponential model
class ExponentialModel(FailureModel):
    def __init__(self, mttf: int):
        self.MTTF=mttf
        self.lambda0=1/mttf
    
    # hazard function
    def hazard_func(self, time: int, delta_time: int) -> float:
        return self.lambda0
    
    # step probability function
    def step_prob_func(self, time: int, delta_time: int) -> float:
        return 1 - np.exp(-self.lambda0*delta_time)
    