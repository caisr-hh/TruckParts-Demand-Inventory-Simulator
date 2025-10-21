from typing import List, Dict, Tuple, Any, Optional, Iterable
import truck as truck_mod
import numpy as np

# Truck class
import importlib
importlib.reload(truck_mod)
Truck = truck_mod.Truck

# list of available failure model
FAILURE_MODEL = [
    "weibull",
    "log-logistic"
]

class Dealer:
    def __init__(self, seed:int, dealer_id: str, n_trucks: int, n_parts: int):
        self.rng = np.random.default_rng(seed)
        self.dealer_id = dealer_id
        self.PARTS_DICT = self.generate_parts_list(n_parts=n_parts, 
                                                   FAILURE_MODEL=FAILURE_MODEL)
        self.MEDIAN_TIME = self.determine_median_time()
        self.K_PARAM = self.uniform_determine_k_parameter()
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
                K_PARAM=self.K_PARAM,
                part_setting="RANDOM"
            ))
        return fleet
    
    # determine median of event occurence time, which is the time at which the failure peobability becomes 50%.
    def determine_median_time(self):
        MEDIAN_TIME={}
        for part_type in self.PARTS_DICT.keys():
            flat_median = self.rng.integers(100, 365)
            hard_median = int(flat_median/2)
            MEDIAN_TIME[part_type] = {"FLAT":flat_median, "HARD":hard_median}
        return MEDIAN_TIME

    # uniformly determine k parameter for each part (uniform(kl,ku))
    def uniform_determine_k_parameter(self):
        K_PARAM={}
        for part_type in self.PARTS_DICT.keys():
            model_kind = self.PARTS_DICT[part_type]["failure_model"]
            # Exponential model
            if model_kind == "exponential":
                K_PARAM[part_type] = {"FLAT":0, "HARD":0}
            # Weibull model
            elif model_kind == "weibull":
                # FLAT
                k_flat = self.rng.random()*(3.0-1.0) + 1.0
                # HARD
                k_hard = self.rng.random()*(0.8-0.4) + 0.4
                
                K_PARAM[part_type] = {"FLAT":k_flat, "HARD":k_hard}

            # Log-logistic model
            elif model_kind == "log-logistic":
                # FLAT
                k_flat = self.rng.random()*(3.0-1.0) + 1.0
                # HARD
                k_hard = k_flat * self.rng.random()*(2.0-1.2) + 1.2

                K_PARAM[part_type] = {"FLAT":k_flat, "HARD":k_hard}

            # Gompertz model
            elif model_kind == "gompertz":
                # FLAT
                k_flat = self.rng.random()*(0.03-0.005) + 0.005
                # HARD
                k_hard = self.rng.random()*(0.1-0.03) + 0.03

                K_PARAM[part_type] = {"FLAT":k_flat, "HARD":k_hard}

        return K_PARAM


    # determine k parameter for each part (norm(const., 0.1))
    def constant_determine_k_parameter(self):
        K_PARAM={}
        for part_type in self.PARTS_DICT.keys():
            model_kind = self.PARTS_DICT[part_type]["failure_model"]
            # Exponential model
            if model_kind == "exponential":
                K_PARAM[part_type] = {"FLAT":0, "HARD":0}
            # Weibull model
            elif model_kind == "weibull":
                # FLAT
                kr_flat = 2.0
                v = self.rng.normal(np.log(kr_flat), 0.1)
                k_flat = np.exp(v)
                while k_flat <= 1:
                    v = self.rng.normal(np.log(kr_flat), 0.1)
                    k_flat = np.exp(v)

                # HARD
                kr_hard = 0.8
                v = self.rng.normal(np.log(kr_hard), 0.05)
                k_hard = np.exp(v)
                while k_hard <= 0 or k_hard > 1:
                    v = self.rng.normal(np.log(kr_hard), 0.05)
                    k_hard = np.exp(v)
                
                K_PARAM[part_type] = {"FLAT":k_flat, "HARD":k_hard}

            # Log-logistic model
            elif model_kind == "log-logistic":
                # FLAT
                kr_flat = 2.5
                v = self.rng.normal(np.log(kr_flat), 0.1)
                k_flat = np.exp(v)
                while k_flat <= 1:
                    v = self.rng.normal(np.log(kr_flat), 0.1)
                    k_flat = np.exp(v)
                
                # HARD
                k_hard = k_flat * 1.5

                K_PARAM[part_type] = {"FLAT":k_flat, "HARD":k_hard}

            # Gompertz model
            elif model_kind == "gompertz":
                # FLAT
                kr_flat = 0.4
                v = self.rng.normal(np.log(kr_flat), 0.1)
                k_flat = np.exp(v)
                while k_flat <= 0:
                    v = self.rng.normal(np.log(kr_flat), 0.1)
                    k_flat = np.exp(v)
                
                # HARD
                k_hard = k_flat * 1.5

                K_PARAM[part_type] = {"FLAT":k_flat, "HARD":k_hard}

        return K_PARAM
    
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