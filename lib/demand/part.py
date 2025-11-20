import numpy as np
import math, random
from typing import Optional
import Event as ev_mod
import FailureModel as fmodel_md

# Event class
import importlib
importlib.reload(ev_mod)
FailureData = ev_mod.FailureData

# Failure Model class
import importlib
importlib.reload(fmodel_md)
Exponential = fmodel_md.ExponentialModel
Weibull = fmodel_md.WeibullModel
LogLogistic = fmodel_md.LogLogisticModel
Gompertz = fmodel_md.GompertzModel


# One Part
class Part:
    def __init__(self, seed: int, dealer_id:str, truck_id: str, model_id: str,
                 part_id: str, part_type: str, usage: str, failure_model: str, 
                 median_time: int, k_param: float, season_rbf):
        # random seed
        self.rng = np.random.default_rng(seed)

        # identifier of holder (dealer)
        self.dealer_id: str = dealer_id

        # identifier of truck
        self.truck_id: str = truck_id
        self.model_id: str = model_id

        # identifier of the part
        self.part_id: str = part_id
        self.part_type: str = part_type

        # usage
        self.usage = usage
        
        # failure model
        self.failure_model = self.make_failure_model(kind=failure_model, median_time=median_time, k_param=k_param, 
                                                     season_rbf=season_rbf)

        # elapsed time after the latest replacement
        self.age = 1
    
    # method: make failure model
    def make_failure_model(self, kind: str, median_time: int, k_param: float, season_rbf: dict):
        child_seed = int(self.rng.integers(0, 2**32 - 1))
        if kind == "exponential":
            base = Exponential(median=median_time, seed=child_seed, 
                               season_rbf=season_rbf)
        elif kind == "weibull":
            base = Weibull(usage=self.usage, median=median_time, k0=k_param, seed=child_seed, 
                           season_rbf=season_rbf)
        elif kind == "log-logistic":
            base = LogLogistic(usage=self.usage, median=median_time, k0=k_param, seed=child_seed, 
                               season_rbf=season_rbf)
        elif kind == "gompertz":
            base = Gompertz(usage=self.usage, median=median_time, k0=k_param, seed=child_seed, 
                            season_rbf=season_rbf)
        return base


    # update parameters according to the operating conditions
    def update_params(self):
        pass

    # evaluate failure model
    def evaluate_failure(self, time: int, delta_time: int, truck_age: int, yearofday:int, days_in_year:int):
        failure_prob = self.failure_model.step_prob_func(time = self.age, delta_time = delta_time, yearofday=yearofday, days_in_year=days_in_year)
        # failure occurs:
        if self.rng.random() < failure_prob:
            ev = FailureData(
                time=time,
                dealer_id=self.dealer_id,
                truck_id=self.truck_id,
                model_id=self.model_id,
                truck_age=truck_age,
                part_id=self.part_id,
                part_type=self.part_type,
                part_age=self.age,
                failure=1
            )             
            self.reset_age()
            return ev
        
        # failure doesn't occur
        ev = FailureData(
            time=time,
            dealer_id=self.dealer_id,
            truck_id=self.truck_id,
            model_id=self.model_id,
            truck_age=truck_age,
            part_id=self.part_id,
            part_type=self.part_type,
            part_age=self.age,
            failure=0
        )
        return ev

    # reset the elapsed time due to replacement
    def reset_age(self) -> None:
        self.age = 0


