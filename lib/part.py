import numpy as np
import math, random
from typing import Optional
import Event as ev_mod

# Event class
import importlib
importlib.reload(ev_mod)
DemandEvent = ev_mod.DemandEvent


# One Part
class Part:
    def __init__(self, part_id: str, part_type: str, 
                 mttf: int = 100):
        # identifier of the part
        self.part_id: str = part_id
        self.part_type: str = part_type
        
        # failure model paramters
        self.mttf: int = mttf   # MTTF (days)
        self.lambda0: float = 1.0 / self.mttf

        # elapsed time after the latest replacement
        self.age = 0
    
    # update parameters according to the operating conditions
    def update_params(self):
        pass

    # evaluate failure model
    def evaluate_failure(self, time: int, delta_time: int, truck_id: str,
                         model_id: str, truck_age: int):
        # failure_prob = self.step_prob_func(delta_time=delta_time)
        failure_prob = self.hazard_func()
        # if failure occurs:
        if np.random.uniform() < failure_prob:
            ev = DemandEvent(
                time=time,
                truck_id=truck_id,
                model_id=model_id,
                truck_age=truck_age,
                part_id=self.part_id,
                part_type=self.part_type,
                part_age=self.age
            )            
            self.reset_age()
            return ev
        return None

    # reset the elapsed time due to replacement
    def reset_age(self) -> None:
        self.age = 0
    
    # hazard function: exponential model
    def hazard_func(self):
        p = self.lambda0
        return p

    # evaluation with step probability (conditional probability)
    def step_prob_func(self, delta_time: int = 1):
        p = 1 - np.exp(-self.lambda0*delta_time)
        return p
    

