import numpy as np
import math, random
from typing import Optional

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
    def evaluate_failure(self):
        pass

    # reset the elapsed time due to replacement
    def reset_life(self) -> None:
        self.age = 0
    

