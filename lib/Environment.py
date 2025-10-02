from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Iterable
from truck import Truck


@dataclass
class SimulationConfig:
    start_time: datetime    # simulation start time 
    total_time: int         # total simulation time
    delta_time: int         # time increment step in each iteration


class Simulator:
    def __init__(self, config: SimulationConfig, n_trucks: int):
        self.config = config
        self.fleet: List[Truck] = self.generate_fleet(n_trucks)
        self.events: list = []

    # generate multiple trucks fleet
    def generate_fleet(self, n_trucks: int, model_id: str = "M0", 
                    id_prefix: str = "T", 
                    PART_LIST: Optional[Iterable[Tuple[str,int]]] = None,
                    MTTF_DAYS: Optional[Dict[str, float]] = None) -> List[Truck]:
        fleet: List[Truck] = []
        for i in range(n_trucks):
            truck_id = f"{id_prefix}{i:03d}"    # the truck indentifier
            fleet.append(Truck(
                truck_id=truck_id,
                model_id=model_id,
                auto_part_setting=True,
                PART_LIST=PART_LIST,
                MTTF_DAYS=MTTF_DAYS
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

    
    


