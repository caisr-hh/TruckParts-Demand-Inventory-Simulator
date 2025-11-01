from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Iterable
import dealer as dealer_mod

# Dealer class
import importlib
importlib.reload(dealer_mod)
Dealer = dealer_mod.Dealer

# from truck import Truck

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
                 n_dealers: int, season_engine, season_strategy: str):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.dealer_failure_model: dict = {}    # for recoding
        self.season_engine = season_engine
        self.seasonality = season_strategy
        self.dealer_list = self.generate_dealer(n_dealers=n_dealers)
        self.events: list = []
    
    # generate dealers
    def generate_dealer(self, n_dealers: int, id_prefix:str = "D"):
        dealer_list: List[Dealer] = []
        for i in range(n_dealers):
            if i%2 == 0:
                loc = "southern"
            else:
                loc = "northern"
            dealer_id = f"{id_prefix}{i:02d}"
            child_seed = int(self.rng.integers(0, 2**32 - 1))
            # n_trucks = int(self.rng.integers(10, 30))
            # n_parts = int(self.rng.integers(25, 30))
            n_trucks = 30
            n_parts = 4
            dealer_list.append(Dealer(
                seed=child_seed,
                dealer_id=dealer_id,
                n_trucks=n_trucks,
                n_parts=n_parts,
                season_engine=self.season_engine,
                seasonality=self.seasonality,
                location = loc
            ))
            self.dealer_failure_model[dealer_id] = dealer_list[-1].PARTS_DICT
        return dealer_list
    
    def run(self):
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

            # increment elapse time
            time += self.config.delta_time
        return self.events
