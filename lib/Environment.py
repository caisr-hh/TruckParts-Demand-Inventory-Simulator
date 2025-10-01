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
    def __init__(self, config: SimulationConfig,
                  fleet: Optional[List[Truck]] = None):
        self.config = config
        self.fleet: List[Truck] = fleet or []
        self.events: list = []
    
    def run(self):
        day = 0     # current date (elapesed day)
        while day < self.config.total_time:
            for truck in self.fleet:
                # evaluate part failure 
                evs = truck.checkup_parts(day=day, delta_t=self.config.delta_time)
                self.events.extend(evs)

                # increment truck age
                truck.increment_age(delta_t=self.config.delta_time)

            # increment elapse time
            day += self.config.delta_time
        return self.events

    
    


