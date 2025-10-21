from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

# global variable
fluctuate = False    # indicate whether the paramter values fluctuate for each part

# abstract class
class FailureModel(ABC):
    # caluculate parameter value inversely from the median
    @ abstractmethod
    def calc_params_val(self):
        ...

    # hazard function
    @ abstractmethod
    def hazard_func(self, time: int) -> float:
        ...
    
    # step probability function (conditional probability)
    @ abstractmethod
    def step_prob_func(self, time: int, delta_time: int) -> float:
        ...


# Exponential model
class ExponentialModel(FailureModel):
    def __init__(self, median: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.median = median
        self.calc_params_val()
    
    # caluculate parameter value inversely from the median
    def calc_params_val(self):
        self.lambda0 = np.log(2)/self.median
    
    # hazard function
    def hazard_func(self, time: int) -> float:
        return self.lambda0
    
    # step probability function (conditional probability)
    def step_prob_func(self, time: int, delta_time: int) -> float:
        return 1 - np.exp(-self.lambda0*delta_time)

# Weibull Model
class WeibullModel(FailureModel):
    def __init__(self, usage: str, median: int, k0: float, seed: int):
        self.rng = np.random.default_rng(seed)
        self.median = median
        self.usage = usage
        self.k0 = k0
        self.calc_params_val()
    
    # caluculate parameter value inversely from the median
    def calc_params_val(self):
        # fluctuate for each part
        if fluctuate:
            sig = 0.01
            if self.usage == "FLAT":
                v = self.rng.normal(np.log(self.k0), sig)
                k0_ = np.exp(v)
                while k0_ < 1:
                    v = self.rng.normal(np.log(self.k0), sig)
                    k0_ = np.exp(v)
                self.k0 = k0_
            
            elif self.usage == "HARD":
                v = self.rng.normal(np.log(self.k0), sig)
                k0_ = np.exp(v)
                while k0_<=0 or k0_>1:
                    v = self.rng.normal(np.log(self.k0), sig)
                    k0_ = np.exp(v)
                self.k0 = k0_

        self.lambda0 = (np.log(2)**(1/self.k0))/self.median

    # hazard function
    def hazard_func(self, time):
        return self.k0*self.lambda0*(time**(self.k0-1))
    
    # step probability function (conditional probability)
    def step_prob_func(self, time, delta_time):
        return 1 - np.exp(self.lambda0*(time**self.k0) - self.lambda0*((time+delta_time)**self.k0))

# Log-logistic Model
class LogLogisticModel(FailureModel):
    def __init__(self, usage: str, median: int, k0: float, seed: int):
        self.rng = np.random.default_rng(seed)
        self.median = median
        self.usage = usage
        self.k0 = k0
        self.calc_params_val()
    
    # caluculate parameter value inversely from the median
    def calc_params_val(self):
        # fluctuate for each part
        if fluctuate:
            sig = 0.1
            v = self.rng.normal(np.log(self.k0), sig)
            k0_ = np.exp(v)
            while k0_ <= 1:
                v = self.rng.normal(np.log(self.k0), sig)
                k0_ = np.exp(v)
            self.k0 = k0_  
        self.lambda0 = 1/(self.median**self.k0)
    
    # hazard function
    def hazard_func(self, time):
        numerator = self.k0*self.lambda0*(time**(self.k0-1))
        denominator = 1+self.lambda0*(time**self.k0)
        return numerator/denominator 
    
    # step probability function (conditional probability)
    def step_prob_func(self, time, delta_time):
        numerator = 1+self.lambda0*(time**self.k0)
        denominator = 1+self.lambda0*((time+delta_time)**self.k0)
        return 1-numerator/denominator

# Gompertz Model
class GompertzModel(FailureModel):
    def __init__(self, usage: str, median: int, k0: float, seed: int):
        self.rng = np.random.default_rng(seed)
        self.median = median
        self.usage = usage
        self.k0 = k0
        self.calc_params_val()
    
    # caluculate parameter value inversely from the median
    def calc_params_val(self):
        # fluctuate for each part
        if fluctuate:
            sig = 0.1
            v = self.rng.normal(np.log(self.k0), sig)
            k0_ = np.exp(v)
            while k0_ <= 0:
                v = self.rng.normal(np.log(self.k0), sig)
                k0_ = np.exp(v)
            self.k0 = k0_  
        self.lambda0 = (1/self.median) * np.log(1 + np.log(2)/self.k0)
    
    # hazard function
    def hazard_func(self, time):
        return self.k0*self.lambda0*np.exp(self.lambda0*time)
    
    # step probability function (conditional probability)
    def step_prob_func(self, time, delta_time):
        return 1-np.exp(self.k0*(np.exp(self.lambda0*time)
                                     -np.exp(self.lambda0*(time+delta_time))))




# # abstract class
# class FailureModel(ABC):
#     # hazard function
#     @ abstractmethod
#     def hazard_func(self, time: int) -> float:
#         ...
    
#     # step probability function (conditional probability)
#     @ abstractmethod
#     def step_prob_func(self, time: int, delta_time: int) -> float:
#         ...


# # Exponential model
# class ExponentialModel(FailureModel):
#     def __init__(self, mttf: int):
#         self.MTTF=mttf
#         self.lambda0=1/mttf
    
#     # hazard function
#     def hazard_func(self, time: int) -> float:
#         return self.lambda0
    
#     # step probability function (conditional probability)
#     def step_prob_func(self, time: int, delta_time: int) -> float:
#         return 1 - np.exp(-self.lambda0*delta_time)

# # Weibull Model
# class WeibullModel(FailureModel):
#     def __init__(self, lambda0: float, alpha0: float):
#         self.lambda0 = lambda0
#         self.alpha0 = alpha0
    
#     # hazard function
#     def hazard_func(self, time):
#         return self.alpha0*self.lambda0*(time**(self.alpha0-1))
    
#     # step probability function (conditional probability)
#     def step_prob_func(self, time, delta_time):
#         return 1 - np.exp(self.lambda0*(time**self.alpha0) - self.lambda0*((time+delta_time)**self.alpha0))

# # Log-logistic Model
# class LogLogisticModel(FailureModel):
#     def __init__(self, lambda0: float, alpha0: float):
#         self.lambda0 = lambda0
#         self.alpha0 = alpha0
    
#     # hazard function
#     def hazard_func(self, time):
#         numerator = self.alpha0*self.lambda0*(time**(self.alpha0-1))
#         denominator = 1+self.lambda0*(time**self.alpha0)
#         return numerator/denominator 
    
#     # step probability function (conditional probability)
#     def step_prob_func(self, time, delta_time):
#         numerator = 1+self.lambda0*(time**self.alpha0)
#         denominator = 1+self.lambda0*((time+delta_time)**self.alpha0)
#         return 1-numerator/denominator

# # Gompertz Model
# class GompertzModel(FailureModel):
#     def __init__(self, lambda0: float, alpha0: float):
#         self.lambda0 = lambda0
#         self.alpha0 = alpha0
    
#     # hazard function
#     def hazard_func(self, time):
#         return self.alpha0*self.lambda0*np.exp(self.lambda0*time)
    
#     # step probability function (conditional probability)
#     def step_prob_func(self, time, delta_time):
#         return 1-np.exp(self.alpha0*(np.exp(self.lambda0*time)
#                                      -np.exp(self.lambda0*(time+delta_time))))
