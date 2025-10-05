from datetime import datetime, timedelta
from dataclasses import dataclass
import random
from typing import List, Dict, Tuple, Any, Optional, Iterable
from truck import Truck

# list of available failure model
FAILURE_MODEL = [
    "exponential",
    "weibull",
    "log-logistic",
    "gompertz"
]


@dataclass
class SimulationConfig:
    start_time: datetime    # simulation start time 
    total_time: int         # total simulation time
    delta_time: int         # time increment step in each iteration


class Simulator:
    def __init__(self, config: SimulationConfig, n_trucks: int, 
                 n_parts: int):
        self.config = config
        common_parts = self.generate_common_parts(n_parts=n_parts, FAILURE_MODEL=FAILURE_MODEL)
        self.fleet: List[Truck] = self.generate_fleet(n_trucks=n_trucks, n_parts=n_parts, PART_DICT=common_parts)
        self.PART_INFO = common_parts
        self.events: list = []

    # ganerate parts list
    def generate_common_parts(self, n_parts: int, FAILURE_MODEL: list):
        COMMON_PART = {}
        for i in range(n_parts):
            # numbering part type and attached failure model
            COMMON_PART["type"+str(i)] = {"failure_model":random.choice(FAILURE_MODEL)} 
        return COMMON_PART

    # generate multiple trucks fleet
    def generate_fleet(self, n_trucks: int, n_parts: int, PART_DICT: dict,
                       model_id: str = "M0", id_prefix: str = "T") -> List[Truck]:
        fleet: List[Truck] = []
        for i in range(n_trucks):
            truck_id = f"{id_prefix}{i:03d}"    # the truck indentifier
            fleet.append(Truck(
                truck_id=truck_id,
                model_id=model_id,
                PART_DICT=PART_DICT,
                part_setting="RANDOM"
            ))
        return fleet
    
    def run(self):
        time = 0     # current time (elapesed time)
        while time < self.config.total_time:
            for truck in self.fleet:
                # evaluate part failure 
                evs = truck.checkup_parts(time=time, delta_time=self.config.delta_time)
                self.events.extend(evs)

                # increment truck age
                truck.increment_age(delta_time=self.config.delta_time)

            # increment elapse time
            time += self.config.delta_time
        return self.events

    
    


