from typing import List, Dict, Tuple, Any, Optional, Iterable
import part as part_mod
import numpy as np

# part class
import importlib
importlib.reload(part_mod)
Part = part_mod.Part

# dictionary of failure model fixed parameters
MODEL_FIXEDPARAMS = {
    "exponential": {"MTTF": 100},
    "weibull": {"lambda0": 1/200, "alpha0": 1.5},
    "log-logistic": {"lambda0": 1/200, "alpha0": 2.5},
    "gompertz": {"lambda0": 1/100, "alpha0": 1/1000}
}


# allocate default part id
def mk_default_part_id(truck_id: str, part_type: str, idx: int) -> str:
    return f"{truck_id}-{part_type}-{idx:03d}"

# allocate part id
def mk_part_id(truck_id: str, part_type: int) -> str:
    return f"{truck_id}-{part_type}"


class Truck:
    def __init__(self, seed: int, dealer_id: str, truck_id: str, model_id: str,
                 PARTS_DICT: dict, usage: str, MEDIAN_TIME: dict,
                 K_PARAM: dict, part_setting: str = "RANDOM"):
        # random seed
        self.rng = np.random.default_rng(seed) 

        # identifier of holder
        self.dealer_id: str = dealer_id

        # identifier of this truck
        self.truck_id: str = truck_id
        self.model_id: str = model_id
        
        # elapsed time after the last replacement of either parts
        self.age: int = 1

        # usage of the truck
        self.usage: str = usage
        
        # parts constituting this truck
        self.parts: List[Part] = []
        self.attach_part(PARTS_DICT=PARTS_DICT, MEDIAN_TIME=MEDIAN_TIME, K_PARAM=K_PARAM)
    

    # attach parts to the truck
    def attach_part(self, PARTS_DICT: dict, MEDIAN_TIME: dict, K_PARAM: dict):
        for part_type in PARTS_DICT.keys():
            # part information
            part_id = mk_part_id(self.truck_id, part_type)

            # failure model
            model_kind = PARTS_DICT[part_type]["failure_model"]
            season_rbf = PARTS_DICT[part_type]["season_rbf"]
            median_time = MEDIAN_TIME[part_type][self.usage]
            k_param = K_PARAM[part_type][self.usage]

            # attach part
            part_seed = int(self.rng.integers(0, 2**32 - 1))
            self.parts.append(Part(seed=part_seed, 
                                dealer_id=self.dealer_id,
                                truck_id=self.truck_id,
                                model_id=self.model_id,
                                part_id=part_id, 
                                part_type=part_type,
                                usage=self.usage,
                                failure_model=model_kind,
                                median_time=median_time,
                                k_param=k_param,
                                season_rbf=season_rbf))

    
    # attach parts to the truck
    def attach_random_part(self, PART_DICT: dict, MODEL_PARAMS: dict):
        for part_type in PART_DICT.keys():
            # part information
            part_id = mk_part_id(self.truck_id, part_type)
            model_kind = PART_DICT[part_type]["failure_model"]
            
            # attach part
            part_seed = int(self.rng.integers(0, 2**32 - 1))
            self.parts.append(Part(seed=part_seed, 
                                dealer_id=self.dealer_id,
                                truck_id=self.truck_id,
                                model_id=self.model_id,
                                part_id=part_id, 
                                part_type=part_type, 
                                failure_model={"kind": model_kind, "params":MODEL_PARAMS[model_kind]}))

    # update operating conditions
    def update_conditions(self):
        pass
    
    # increment truck age and its parts ages
    def increment_age(self, delta_time: int):
        self.age += delta_time
        for part in self.parts:
            part.age += delta_time

    # daily checkup on each part 
    def checkup_parts(self, time: int, delta_time: int, yearofday: int, days_in_year: int):
        events = []
        for part in self.parts:
            # print(time)
            ev = part.evaluate_failure(time=time, delta_time=delta_time, truck_age=self.age, yearofday=yearofday, days_in_year=days_in_year)
            
            events.append(ev)
        return events


    