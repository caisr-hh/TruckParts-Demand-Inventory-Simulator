from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Iterable
import dealer as dealer_md
import demand_management as demand_md

# Dealer class
import importlib
importlib.reload(dealer_md)
Dealer = dealer_md.Dealer

# DemandDataManager class
importlib.reload(demand_md)
DemandManager = demand_md.DataManager

@dataclass
class SimulationConfig:
    start_time: datetime    # simulation start time
    end_time: datetime
    delta_time: int         # time increment step in each iteration
    total_time: int = field(init=False)       # total simulation time

    def __post_init__(self):
        diff = self.end_time - self.start_time   
        days = diff.days                   
        self.total_time = days // self.delta_time


class Simulator:
    def __init__(self, config: SimulationConfig, seed: int, 
                 n_dealers: int, n_truck_range: list, n_part_range: list):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.dealer_list = self.generate_dealer(n_dealers=n_dealers, 
                                                n_truck_range=n_truck_range,
                                                n_part_range=n_part_range)
        self.events: list = []
    
    # generate dealers
    def generate_dealer(self, n_dealers: int, n_truck_range: list, 
                        n_part_range: list, id_prefix:str = "D"):
        dealer_list: List[Dealer] = []
        for i in range(n_dealers):
            dealer_id = f"{id_prefix}{i:02d}"
            child_seed = int(self.rng.integers(0, 2**32 - 1))
            n_trucks = int(self.rng.integers(n_truck_range[0], n_truck_range[1]))
            n_parts = int(self.rng.integers(n_part_range[0], n_part_range[1]))
            dealer_list.append(Dealer(
                seed=child_seed,
                dealer_id=dealer_id,
                n_trucks=n_trucks,
                n_parts=n_parts,
                seasonality="randomRBF",
                drift_type="None"
            ))
        return dealer_list
    
    def run(self, visualize="None"):
        time = 1     # current time (elapesed time)
        while time <= self.config.total_time:
            # time
            yearofday = int((self.config.start_time + timedelta(days=time)).timetuple().tm_yday)
            month = (self.config.start_time + timedelta(days=time)).month
            year = (self.config.start_time + timedelta(days=time)).year
            days_in_year = datetime(year, 12, 31).timetuple().tm_yday
            for dealer in self.dealer_list:
                # management of own trucks
                evs = dealer.manage_trucks(time=time, delta_time=self.config.delta_time, yearofday=yearofday, days_in_year=days_in_year)
                self.events.extend(evs)

                # apply drift
                dealer.apply_drifts(start_date=self.config.start_time,
                                    current_date=self.config.start_time + timedelta(days=time))

            # increment elapse time
            time += self.config.delta_time
        
        # organize demand data
        demand_manage = DemandManager(self.events, self.config.start_time)
        demand_manage.save_dealer_info(self.dealer_list)

        return self.events

