from typing import List, Dict, Tuple, Any, Optional, Iterable
import truck as truck_mod
import Parameter as param_mod
import numpy as np
from datetime import datetime

# Truck class
import importlib
importlib.reload(truck_mod)
Truck = truck_mod.Truck

# Parameter class
importlib.reload(param_mod)
RBFPRAM = param_mod.RBFPARAM

# list of available failure model
FAILURE_MODEL = [
    "weibull",
    "log-logistic",
    "gompertz",
    "exponential"
]

# list of location 
LOCATION = [
    "southern",
    "northern"
]

# dictionary of RBF information
RBF = {
    "southern": {"summer":{"A":5,"c":"2025/8/1","w":50}, "winter":{"A":2.5,"c":"2025/1/15","w":30}},
    "northern": {"summer":{"A":2.5,"c":"2025/8/1","w":30}, "winter":{"A":5,"c":"2025/1/15","w":50}}
}

class Dealer:
    def __init__(self, seed:int, dealer_id: str, n_trucks: int, n_parts: int,
                 seasonality: str, drift_type: str):
        # random seed
        self.rng = np.random.default_rng(seed)

        # dealer information
        self.dealer_id = dealer_id                                  # id
        self.location = self.determine_location(LOCATION=LOCATION)  # location
        self.seasonality = seasonality                              # seasonality handler
        self.drift_type = drift_type                                # concept drift
        self.set_drift_strategy()

        # holding parts 
        self.n_parts = n_parts
        self.PARTS_DICT, self.MEDIAN_TIME, self.K_PARAM = self.generate_parts_list(FAILURE_MODEL=FAILURE_MODEL)
        
        # holding trucks
        self.n_trucks = n_trucks
        self.fleet: List[Truck] = self.generate_fleet(model_id=dealer_id)

    # generate multiple trucks fleet
    def generate_fleet(self, model_id: str, id_prefix: str = "T") -> List[Truck]:
        fleet: List[Truck] = []
        for i in range(self.n_trucks):
            # id
            truck_id = f"{id_prefix}{i:03d}"

            # usage
            usage = self.determine_usage()

            # generation
            child_seed = int(self.rng.integers(0, 2**32 - 1))
            fleet.append(Truck(
                seed=child_seed,
                dealer_id=self.dealer_id,
                truck_id=truck_id,
                model_id=model_id,
                PARTS_DICT=self.PARTS_DICT,
                usage=usage,
                MEDIAN_TIME=self.MEDIAN_TIME,
                K_PARAM=self.K_PARAM,
                part_setting="RANDOM"
            ))
        return fleet

    # method: generate holding parts
    def generate_parts_list(self, FAILURE_MODEL: list):
        PARTS_DICT = {}
        MEDIAN_TIME = {}
        K_PARAM = {}
        for i in range(self.n_parts):
            # part type
            part_type = "type"+str(i)

            # failure model (attach)
            model = self.rng.choice(FAILURE_MODEL)

            # seasonality (attach)
            rbf_param, season_type = self.determine_seasonality_hazard_model(strategy=self.seasonality, i_part=i)

            # median lifetime (determine)
            median_lifetime = self.determine_median_lifetime()
            MEDIAN_TIME[part_type] = median_lifetime

            # parameter k (determine)
            parameter_k = self.determine_k_parameter(model)
            K_PARAM[part_type] = parameter_k

            # part information
            PARTS_DICT[part_type] = {"failure_model":model, "season_rbf":rbf_param,
                                         "season_type": season_type, "location": self.location,
                                         "flat_medtime": median_lifetime["FLAT"], 
                                         "hard_medtime": median_lifetime["HARD"]}
        return PARTS_DICT, MEDIAN_TIME, K_PARAM
    
    # method: determine median of demand occurence time
    def determine_median_lifetime(self):
        flat_median = self.rng.integers(100, 150)
        hard_coef = self.rng.random()*(9/10-1/2) + 1/2
        # hard_median = int(flat_median*2/3)
        hard_median = hard_coef*flat_median
        median_lifetime = {"FLAT":flat_median, "HARD":hard_median}
        return median_lifetime
    
    # method: determine k parameter 
    def determine_k_parameter(self, model_kind):
        parameter_k={}
        # Exponential model
        if model_kind == "exponential":
            parameter_k = {"FLAT":0, "HARD":0}
        
        # Weibull model
        elif model_kind == "weibull":
            # FLAT
            k_flat = self.rng.random()*(2.0-1.5) + 1.5
            # HARD
            k_hard = self.rng.random()*(0.7-0.4) + 0.4
            parameter_k = {"FLAT":k_flat, "HARD":k_hard}

        # Log-logistic model
        elif model_kind == "log-logistic":
            # FLAT
            k_flat = self.rng.random()*(3.0-1.0) + 1.0
            # HARD
            k_hard = k_flat * self.rng.random()*(2.0-1.2) + 1.2
            parameter_k = {"FLAT":k_flat, "HARD":k_hard}

        # Gompertz model
        elif model_kind == "gompertz":
            # FLAT
            k_flat = self.rng.random()*(0.03-0.005) + 0.005
            # HARD
            k_hard = self.rng.random()*(0.1-0.03) + 0.03
            parameter_k = {"FLAT":k_flat, "HARD":k_hard}

        return parameter_k
    
    # add trucks in fleets
    def add_truck_to_fleet(self, n_add: int, id_prefix: str = "T"):
        new_trucks = []
        for i in range(n_add):
            truck_id = f"{id_prefix}{i+len(self.fleet):03d}"
            usage = self.determine_usage()
            child_seed = int(self.rng.integers(0, 2**32 - 1))
            trucki = Truck(
                seed=child_seed,
                dealer_id=self.dealer_id,
                truck_id=truck_id,
                model_id=self.dealer_id,
                PARTS_DICT=self.PARTS_DICT,
                usage=usage,
                MEDIAN_TIME=self.MEDIAN_TIME,
                K_PARAM=self.K_PARAM,
                part_setting="RANDOM"
            )
            new_trucks.append(trucki)
        self.fleet.extend(new_trucks)
    
    # method: set drift strategy
    def set_drift_strategy(self, strategy: str = "fix"):
        # sudden drift
        if self.drift_type == "sudden" or self.drift_type == "both":
            self._sudden_applied_months = set()
            # fix
            if strategy == "fix":
                self.sudden_drift_months = [3, 12, 22, 26, 30]
            
        # slow drift
        if self.drift_type == "slow" or self.drift_type == "both":
            # fix
            if strategy == "fix":
                self.slow_start_month = 0
                self.slow_start_month = 12

    # method: apply drift
    def apply_drifts(self, start_date: datetime, current_date: datetime):
        month_no = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month)
        day_of_month = current_date.day

        # None
        if self.drift_type == "None":
            return
        
        # sudden or both
        if self.drift_type in ("sudden", "both"):
            if day_of_month == 1 and month_no in self.sudden_drift_months and month_no not in self._sudden_applied_months:
                n_add = int(len(self.fleet)*0.5)
                self.add_truck_to_fleet(n_add=n_add)
                self._sudden_applied_months.add(month_no)
        
        # slow or both
        if self.drift_type in ("slow", "both"):
            if month_no >= self.slow_start_month and day_of_month == 1:
                self.add_truck_to_fleet(n_add=1)


    # method: determine dealer location
    def determine_location(self, LOCATION: list):
        return self.rng.choice(LOCATION)
    
    
    # method: determine seasonalilty characteristic for hazard model
    def determine_seasonality_hazard_model(self, strategy: str, i_part: int):
        #- Season Distribution -#
        # random distribution of seasonal type
        if strategy == "randomRBF":
            seasonal_type = self.rng.choice(["no-season", "both", "winter", "summer"])
        
        # comprehensive distribution of seasonal type
        elif strategy == "comprehensiveRBF":    
            # season type
            season_list = ["both", "summer", "winter", "no-season"]
            key = i_part%4
            seasonal_type = season_list[key]

        #- Setting of RBF Parameters -#
        if seasonal_type == "no-season":    # no-season
            A = [0, 0]
            c = [datetime.strptime(RBF[self.location]["summer"]["c"],"%Y/%m/%d").timetuple().tm_yday, 
                    datetime.strptime(RBF[self.location]["winter"]["c"],"%Y/%m/%d").timetuple().tm_yday]
            w = [RBF[self.location]["summer"]["w"],
                    RBF[self.location]["winter"]["w"]]
            s = len(A)

        elif seasonal_type == "both":       # both season
            A = [RBF[self.location]["summer"]["A"], 
                    RBF[self.location]["winter"]["A"]]
            c = [datetime.strptime(RBF[self.location]["summer"]["c"],"%Y/%m/%d").timetuple().tm_yday, 
                    datetime.strptime(RBF[self.location]["winter"]["c"],"%Y/%m/%d").timetuple().tm_yday]
            w = [RBF[self.location]["summer"]["w"], 
                    RBF[self.location]["winter"]["w"]]
            s = len(A)
        
        elif seasonal_type == "summer":     # summer season
            A = [RBF[self.location]["summer"]["A"], 0]
            c = [datetime.strptime(RBF[self.location]["summer"]["c"],"%Y/%m/%d").timetuple().tm_yday, 
                    datetime.strptime(RBF[self.location]["winter"]["c"],"%Y/%m/%d").timetuple().tm_yday]
            w = [RBF[self.location]["summer"]["w"], 
                    RBF[self.location]["winter"]["w"]]
            s = len(A)
        
        elif seasonal_type == "winter":     # winter season
            A = [0, RBF[self.location]["winter"]["A"]]
            c = [datetime.strptime(RBF[self.location]["summer"]["c"],"%Y/%m/%d").timetuple().tm_yday, 
                    datetime.strptime(RBF[self.location]["winter"]["c"],"%Y/%m/%d").timetuple().tm_yday]
            w = [RBF[self.location]["summer"]["w"], 
                    RBF[self.location]["winter"]["w"]]
            s = len(A)

        rbf_param = RBFPRAM(s, A, c, w)

        return rbf_param, seasonal_type

    # method: determine truck usage randomly
    def determine_usage(self):
        # flat:hard = 1:1
        if self.rng.random()<0.5:
            return "FLAT"
        else:
            return "HARD"
    
    # method: manage holding trucks
    def manage_trucks(self, time:int, delta_time:int, yearofday: int, days_in_year: int):
        events_trucks = []
        for truck in self.fleet:
            # evaluate part failure
            evs = truck.checkup_parts(time=time, delta_time=delta_time, yearofday=yearofday, days_in_year=days_in_year)
            events_trucks.extend(evs)

            # truck is proceeding
            truck.increment_age(delta_time=delta_time)
        return events_trucks 

