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
    def __init__(self, median: int, seed: int, season_rbf: dict):
        self.rng = np.random.default_rng(seed)
        self.median = median
        self.season_rbf = season_rbf
        self.calc_params_val()
    
    # caluculate parameter value inversely from the median
    def calc_params_val(self):
        self.lambda0 = np.log(2)/self.median
    
    # hazard function
    def hazard_func(self, time: int, month: int) -> float:
        return (self.lambda0*np.exp(self.season_param_coef[month]))*self.season_coef[month]
    
    # step probability function (conditional probability)
    def step_prob_func(self, time: int, delta_time: int, month: int) -> float:
        return 1 - np.exp(-(self.lambda0*np.exp(self.season_param_coef[month]))*delta_time)

# Weibull Model
class WeibullModel(FailureModel):
    def __init__(self, usage: str, median: int, k0: float, seed: int, season_rbf):
        self.rng = np.random.default_rng(seed)
        self.median = median
        self.usage = usage
        self.k0 = k0
        self.season_rbf = season_rbf
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
    def hazard_func(self, time, yearofday, days_in_year):
        w = self.rbf_coef_func(yearofday, days_in_year)      # coefficient
        return self.k0*(self.lambda0*np.exp(0))*(time**(self.k0-1))*w
    
    # step probability function (conditional probability)
    def step_prob_func(self, time, delta_time, yearofday, days_in_year):
        w = self.rbf_coef_func(yearofday, days_in_year)      # coefficient
        return 1 - np.exp(w*((self.lambda0)*(time**self.k0) - (self.lambda0)*((time+delta_time)**self.k0)))
    
    # coefficient calculation using RBF
    def rbf_coef_func(self, yearofday, days_in_year):
        f = 0
        s = self.season_rbf.s

        # sum up each rbf using circle coordinate
        for i in range(s):
            # rbf components
            Ai = self.season_rbf.A[i]
            ci = self.season_rbf.c[i]
            wi = self.season_rbf.w[i]
            di = min([abs(ci-yearofday),days_in_year-abs(ci-yearofday)])
            
            # sum up
            f += Ai * np.exp(-(di**2)/((wi**2)))
        
        # rbf
        f = f + 1
        return f

# Log-logistic Model
class LogLogisticModel(FailureModel):
    def __init__(self, usage: str, median: int, k0: float, seed: int, season_rbf):
        self.rng = np.random.default_rng(seed)
        self.median = median
        self.usage = usage
        self.k0 = k0
        self.season_rbf = season_rbf
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
    def hazard_func(self, time, month):
        numerator = self.k0*(self.lambda0*np.exp(self.season_param_coef[month]))*(time**(self.k0-1))
        denominator = 1+(self.lambda0*np.exp(self.season_param_coef[month]))*(time**self.k0)
        return numerator/denominator*self.season_coef[month]
    
    # step probability function (conditional probability)
    def step_prob_func(self, time, delta_time, month):
        numerator = 1+(self.lambda0*np.exp(self.season_param_coef[month]))*(time**self.k0)
        denominator = 1+(self.lambda0*np.exp(self.season_param_coef[month]))*((time+delta_time)**self.k0)
        return 1-(numerator/denominator)**self.season_coef[month]

# Gompertz Model
class GompertzModel(FailureModel):
    def __init__(self, usage: str, median: int, k0: float, seed: int, season_rbf):
        self.rng = np.random.default_rng(seed)
        self.median = median
        self.usage = usage
        self.k0 = k0
        self.season_rbf = season_rbf
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
    def hazard_func(self, time, month):
        return self.k0*(self.lambda0*np.exp(self.season_param_coef[month]))*np.exp((self.lambda0*self.season_param_coef[month])*time)*self.season_coef[month]
    
    # step probability function (conditional probability)
    def step_prob_func(self, time, delta_time, month):
        return 1-np.exp(self.k0*(np.exp((self.lambda0*np.exp(self.season_param_coef[month]))*time)
                                     -np.exp((self.lambda0*np.exp(self.season_param_coef[month]))*(time+delta_time))))

