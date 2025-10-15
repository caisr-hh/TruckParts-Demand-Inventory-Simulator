from typing import List, Dict, Tuple, Any, Optional, Iterable
import truck as truck_mod
import numpy as np

# Truck class
import importlib
importlib.reload(truck_mod)
Truck = truck_mod.Truck

# list of available failure model
FAILURE_MODEL = [
    "exponential",
    "weibull",
    "log-logistic",
    "gompertz"
]

class Dealer:
    def __init__(self, seed:int, dealer_id: str, n_trucks: int, n_parts: int):
        self.rng = np.random.default_rng(seed)
        self.dealer_id = dealer_id
        self.PARTS_DICT = self.generate_parts_list(n_parts=n_parts, 
                                                   FAILURE_MODEL=FAILURE_MODEL)
        self.MEDIAN_TIME = self.determine_median_time(n_parts=n_parts)
        print(self.MEDIAN_TIME)
        self.fleet: List[Truck] = self.generate_fleet(n_trucks=n_trucks, n_parts=n_parts,
                                                      model_id=dealer_id)

    # ganerate unspecific parts list
    def generate_parts_list(self, n_parts: int, FAILURE_MODEL: list):
        PARTS_DICT = {}
        for i in range(n_parts):
            # numbering part type and attached failure model
            model = self.rng.choice(FAILURE_MODEL)
            PARTS_DICT["type"+str(i)] = {"failure_model":model} 
        return PARTS_DICT
    
    # generate multiple trucks fleet
    def generate_fleet(self, n_trucks: int, n_parts: int,
                       model_id: str, id_prefix: str = "T") -> List[Truck]:
        fleet: List[Truck] = []
        for i in range(n_trucks):
            truck_id = f"{id_prefix}{i:03d}"    # the truck indentifier
            usage = self.determine_usage()      # FIX: parameter fixed, others: variable
            # usage = "FIX"
            child_seed = int(self.rng.integers(0, 2**32 - 1))
            fleet.append(Truck(
                seed=child_seed,
                dealer_id=self.dealer_id,
                truck_id=truck_id,
                model_id=model_id,
                PARTS_DICT=self.PARTS_DICT,
                usage=usage,
                MEDIAN_TIME=self.MEDIAN_TIME, 
                part_setting="RANDOM"
            ))
        return fleet
    
    # determine median of event occurence time, which is the time at which the failure peobability becomes 50%.
    def determine_median_time(self, n_parts: int):
        MEDIAN_TIME={}
        for part_type in self.PARTS_DICT.keys():
            flat_median = self.rng.integers(100, 365)
            hard_median = int(flat_median/2)
            MEDIAN_TIME[part_type] = {"FLAT":flat_median, "HARD":hard_median}
        return MEDIAN_TIME

    
    # determine truck usage randomly
    def determine_usage(self):
        # flat:hard = 1:1
        if self.rng.random()<0.5:
            return "FLAT"
        else:
            return "HARD"
    
    # manage holding trucks
    def manage_trucks(self, time:int, delta_time:int):
        events_trucks = []
        for truck in self.fleet:
            # evaluate part failure
            evs = truck.checkup_parts(time=time, delta_time=delta_time)
            events_trucks.extend(evs)

            # truck is proceeding
            truck.increment_age(delta_time=delta_time)
        return events_trucks 