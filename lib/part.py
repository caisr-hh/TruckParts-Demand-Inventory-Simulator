import numpy as np
import math, random
from typing import Optional
from Event import DemandEvent

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
    def evaluate_failure(self, delta_t: int, day: int, truck_id: str,
                         model_id: str, truck_age: int):
        failure_prob = self.step_prob_exp(delta_t=delta_t)
        # if failure occurs:
        if np.random.uniform() < failure_prob:
            ev = DemandEvent(
                day=day,
                truck_id=truck_id,
                model_id=model_id,
                truck_age=truck_age,
                part_id=self.part_id,
                part_type=self.part_type,
                part_age=self.age
            )            
            self.reset_life()
            return ev
        return None

    # reset the elapsed time due to replacement
    def reset_life(self) -> None:
        self.age = 0
    
    # hazard function: exponential model
    def hazard_func_exp(self):
        p = self.lambda0
        return p

    # evaluation with step probability (conditional probability)
    def step_prob_exp(self, delta_t: int = 1):
        p = 1 - np.exp(-self.lambda0*delta_t)
        return p
    

