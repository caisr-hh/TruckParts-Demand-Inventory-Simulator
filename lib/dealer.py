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
    "weibull"
]

# list of location 
LOCATION = [
    "southern",
    "northern"
]

# dictionary of RBF information
RBF = {
    "southern": {"summer":{"A":10,"c":"2025/8/1","w":25}, "winter":{"A":5,"c":"2025/1/15","w":15}},
    "northern": {"summer":{"A":5,"c":"2025/8/1","w":15}, "winter":{"A":10,"c":"2025/1/15","w":25}}
}

class Dealer:
    def __init__(self, seed:int, dealer_id: str, n_trucks: int, n_parts: int,
                 season_engine: str, seasonality: str, location: str):
        self.rng = np.random.default_rng(seed)
        # basic information
        self.dealer_id = dealer_id
        self.location = location

        # seasonality
        self.season_engine = season_engine
        self.seasonality = seasonality

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
            # attach part type id, failure model and seasonal coefficient
            model = self.rng.choice(FAILURE_MODEL)

            rbf_param, season_type = self.determine_seasonality_hazard_model(strategy=self.seasonality)

            PARTS_DICT["type"+str(i)] = {"failure_model":model, "season_rbf":rbf_param
                                         , "season_type": season_type, "location": self.location} 
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
    
    # determine dealer location
    def determine_location(self, LOCATION: list):
        return self.rng.choice(LOCATION)
    
    
    # determine seasonalilty characteristic for hazard model
    def determine_seasonality_hazard_model(self, strategy: str):
        # # no-seasonality
        # if strategy == "None":
        #     Ws = [{1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
        #         7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12:1.0},
        #         "no-season"]
        
        # # randomly chosen constant coefficient
        # elif strategy == "random_constant":
        #     high = 5.0
        #     transition = high/2
        #     normal = 1.0
        #     # winter - summer
        #     Wboth = {1: high, 2: high, 3: transition, 4: normal, 5: normal, 6: transition, 
        #          7: high, 8: high, 9: transition, 10: normal, 11: transition, 12: high}
        #     # winter
        #     Wwinter = {1: high, 2: high, 3: transition, 4: normal, 5: normal, 6: normal, 
        #          7: normal, 8: normal, 9: normal, 10: normal, 11: transition, 12: high}
        #     # summer
        #     Wsummer = {1: normal, 2: normal, 3: normal, 4: normal, 5: normal, 6: transition, 
        #          7: high, 8: high, 9: transition, 10: normal, 11: normal, 12: normal}
        #     Ws = self.rng.choice([[Wboth,"both"], [Wwinter,"winter"], [Wsummer,"summer"]])

        # # RBF based
        # elif strategy == "RBF":
        # season type
        season_type = self.rng.choice(["no-season", "both", "winter", "summer"])

        # no-season
        if season_type == "no-season":
            A = [0, 0]
            c = [datetime.strptime(RBF[self.location]["summer"]["c"],"%Y/%m/%d").timetuple().tm_yday, 
                    datetime.strptime(RBF[self.location]["winter"]["c"],"%Y/%m/%d").timetuple().tm_yday]
            w = [RBF[self.location]["summer"]["w"], 
                    RBF[self.location]["winter"]["w"]]
            s = len(A)
        # both season
        elif season_type == "both":
            A = [RBF[self.location]["summer"]["A"], 
                    RBF[self.location]["winter"]["A"]]
            c = [datetime.strptime(RBF[self.location]["summer"]["c"],"%Y/%m/%d").timetuple().tm_yday, 
                    datetime.strptime(RBF[self.location]["winter"]["c"],"%Y/%m/%d").timetuple().tm_yday]
            w = [RBF[self.location]["summer"]["w"], 
                    RBF[self.location]["winter"]["w"]]
            s = len(A)
        # summer season
        elif season_type == "summer":
            A = [RBF[self.location]["summer"]["A"], 0]
            c = [datetime.strptime(RBF[self.location]["summer"]["c"],"%Y/%m/%d").timetuple().tm_yday, 
                    datetime.strptime(RBF[self.location]["winter"]["c"],"%Y/%m/%d").timetuple().tm_yday]
            w = [RBF[self.location]["summer"]["w"], 
                    RBF[self.location]["winter"]["w"]]
            s = len(A)
        # winter season
        elif season_type == "winter":
            A = [0, RBF[self.location]["winter"]["A"]]
            c = [datetime.strptime(RBF[self.location]["summer"]["c"],"%Y/%m/%d").timetuple().tm_yday, 
                    datetime.strptime(RBF[self.location]["winter"]["c"],"%Y/%m/%d").timetuple().tm_yday]
            w = [RBF[self.location]["summer"]["w"], 
                    RBF[self.location]["winter"]["w"]]
            s = len(A)
        
        rbf_param = RBFPRAM(s, A, c, w)

        return rbf_param, season_type
    

    # determine seasonalilty characteristic for parameter (AFT Model)
    def determine_seasonality_parameter(self, strategy: str):
        beta = [{1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0,
            7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0},
            "no-season"]

        return beta
            
    
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
    def manage_trucks(self, time:int, delta_time:int, yearofday: int, days_in_year: int):
        events_trucks = []
        for truck in self.fleet:
            # evaluate part failure
            evs = truck.checkup_parts(time=time, delta_time=delta_time, yearofday=yearofday, days_in_year=days_in_year)
            events_trucks.extend(evs)

            # truck is proceeding
            truck.increment_age(delta_time=delta_time)
        return events_trucks 



# class Dealer:
#     def __init__(self, seed:int, dealer_id: str, n_trucks: int, n_parts: int,
#                  season_engine: str, seasonality: str):
#         self.rng = np.random.default_rng(seed)
#         self.dealer_id = dealer_id
#         self.season_engine = season_engine
#         self.seasonality = seasonality
#         self.PARTS_DICT = self.generate_parts_list(n_parts=n_parts, 
#                                                    FAILURE_MODEL=FAILURE_MODEL)
#         self.MEDIAN_TIME = self.determine_median_time()
#         self.K_PARAM = self.uniform_determine_k_parameter()
#         self.fleet: List[Truck] = self.generate_fleet(n_trucks=n_trucks, n_parts=n_parts,
#                                                       model_id=dealer_id)

#     # ganerate unspecific parts list
#     def generate_parts_list(self, n_parts: int, FAILURE_MODEL: list):
#         PARTS_DICT = {}
#         for i in range(n_parts):
#             # attach part type id, failure model and seasonal coefficient
#             model = self.rng.choice(FAILURE_MODEL)

#             if self.season_engine=="Hazard":
#                 Ws_list = self.determine_seasonality_hazard_model(strategy=self.seasonality)
#                 Ws, season_type = Ws_list[0], Ws_list[1]
#                 beta_list = self.determine_seasonality_parameter(strategy="None")
#                 beta = beta_list[0]
            
#             # elif self.season_engine=="Parameter":
#             #     Ws_list = self.determine_seasonality_hazard_model(strategy="None")
#             #     Ws = Ws_list[0]
#             #     beta_list = self.determine_seasonality_parameter(strategy=self.seasonality)
#             #     beta, season_type = beta_list[0], beta_list[1]
            
#             # elif self.season_engine=="None":
#             #     Ws_list = self.determine_seasonality_hazard_model(strategy="None")
#             #     Ws = Ws_list[0]
#             #     beta_list = self.determine_seasonality_parameter(strategy="None")
#             #     beta, season_type = beta_list[0], beta_list[1]

#             PARTS_DICT["type"+str(i)] = {"failure_model":model, "season_coef":Ws
#                                          , "season_param_coef":beta, "season_type": season_type} 
#         return PARTS_DICT
    
#     # generate multiple trucks fleet
#     def generate_fleet(self, n_trucks: int, n_parts: int,
#                        model_id: str, id_prefix: str = "T") -> List[Truck]:
#         fleet: List[Truck] = []
#         for i in range(n_trucks):
#             truck_id = f"{id_prefix}{i:03d}"    # the truck indentifier
#             usage = self.determine_usage()      # FIX: parameter fixed, others: variable
#             # usage = "FIX"
#             child_seed = int(self.rng.integers(0, 2**32 - 1))
#             fleet.append(Truck(
#                 seed=child_seed,
#                 dealer_id=self.dealer_id,
#                 truck_id=truck_id,
#                 model_id=model_id,
#                 PARTS_DICT=self.PARTS_DICT,
#                 usage=usage,
#                 MEDIAN_TIME=self.MEDIAN_TIME,
#                 K_PARAM=self.K_PARAM,
#                 part_setting="RANDOM"
#             ))
#         return fleet
    
#     # determine dealer location
#     def determine_location(self, LOCATION: list):
#         return self.rng.choice(LOCATION)
    
    
#     # determine seasonalilty characteristic for hazard model
#     def determine_seasonality_hazard_model(self, strategy: str):
#         # no-seasonality
#         if strategy == "None":
#             Ws = [{1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
#                 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12:1.0},
#                 "no-season"]
        
#         # constant coefficient
#         elif strategy == "constant":
#             high = 5.0
#             transition = high/2
#             normal = 1.0
#             # coefficient
#             Ws = [{1: high, 2: high, 3: transition, 4: normal, 5: normal, 6: transition, 
#                  7: high, 8: high, 9: transition, 10: normal, 11: transition, 12: high},
#                  "both"]
        
#         # randomly chosen constant coefficient
#         elif strategy == "random_constant":
#             high = 5.0
#             transition = high/2
#             normal = 1.0
#             # winter - summer
#             Wboth = {1: high, 2: high, 3: transition, 4: normal, 5: normal, 6: transition, 
#                  7: high, 8: high, 9: transition, 10: normal, 11: transition, 12: high}
#             # winter
#             Wwinter = {1: high, 2: high, 3: transition, 4: normal, 5: normal, 6: normal, 
#                  7: normal, 8: normal, 9: normal, 10: normal, 11: transition, 12: high}
#             # summer
#             Wsummer = {1: normal, 2: normal, 3: normal, 4: normal, 5: normal, 6: transition, 
#                  7: high, 8: high, 9: transition, 10: normal, 11: normal, 12: normal}
#             Ws = self.rng.choice([[Wboth,"both"], [Wwinter,"winter"], [Wsummer,"summer"]])

#         # random sin wave
#         elif strategy == "sin_wave":
#             season_type = self.rng.choice(["both", "summer", "winter"])
#             A = 2
#             if season_type == "both":
#                 T = 12
#                 fai = self.rng.choice([np.pi/2,np.pi/3,np.pi/6])
                
#                 # coefficient
#                 W = {}
#                 for i in range(12):
#                     W[i+1] = 1 + A*np.abs(np.sin((2*np.pi*(i+1))/T+fai))
            
#             elif season_type == "summer":
#                 T = 24
#                 fai = self.rng.choice([0,-np.pi/12,-np.pi/6])

#                 # coefficient
#                 W = {}
#                 for i in range(12):
#                     W[i+1] = 1 + A*np.abs(np.sin((2*np.pi*(i+1))/T+fai))
            
#             elif season_type == "winter":
#                 T = 24
#                 fai = self.rng.choice([-np.pi/2,(5*np.pi)/12,np.pi/3])

#                 # coefficient
#                 W = {}
#                 for i in range(12):
#                     W[i+1] = 1 + A*np.abs(np.sin((2*np.pi*(i+1))/T+fai))
#             Ws = [W, season_type]

#         return Ws
    

#     # determine seasonalilty characteristic for parameter (AFT Model)
#     def determine_seasonality_parameter(self, strategy: str):
#         # no-seasonality
#         if strategy == "None":
#             beta = [{1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0,
#                 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0},
#                 "no-season"]
        
#         # constant coefficient
#         elif strategy == "constant":
#             high = 2.3
#             transition = high/2
#             normal = 0.0
#             # coefficient
#             beta = [{1: high, 2: high, 3: transition, 4: normal, 5: normal, 6: transition, 
#                  7: high, 8: high, 9: transition, 10: normal, 11: transition, 12: high},
#                  "both"]
        
#         # randomly chosen constant coefficient
#         elif strategy == "random_constant":
#             high = 2.3
#             transition = high/2
#             normal = 0.0
#             # winter - summer
#             beta_both = {1: high, 2: high, 3: transition, 4: normal, 5: normal, 6: transition, 
#                  7: high, 8: high, 9: transition, 10: normal, 11: transition, 12: high}
#             # winter
#             beta_winter = {1: high, 2: high, 3: transition, 4: normal, 5: normal, 6: normal, 
#                  7: normal, 8: normal, 9: normal, 10: normal, 11: transition, 12: high}
#             # summer
#             beta_summer = {1: normal, 2: normal, 3: normal, 4: normal, 5: normal, 6: transition, 
#                  7: high, 8: high, 9: transition, 10: normal, 11: normal, 12: normal}
#             beta = self.rng.choice([[beta_both,"both"], [beta_winter,"winter"], [beta_summer,"summer"]])

#         # random sin wave
#         elif strategy == "sin_wave":
#             season_type = self.rng.choice(["both", "summer", "winter"])
#             A = 0.4
#             if season_type == "both":
#                 T = 12
#                 fai = self.rng.choice([np.pi/2,np.pi/3,np.pi/6])
                
#                 # coefficient
#                 beta_val = {}
#                 for i in range(12):
#                     beta_val[i+1] = A*np.abs(np.sin((2*np.pi*(i+1))/T+fai))
            
#             elif season_type == "summer":
#                 T = 24
#                 fai = self.rng.choice([0,-np.pi/12,-np.pi/6])

#                 # coefficient
#                 beta_val = {}
#                 for i in range(12):
#                     beta_val[i+1] = A*np.abs(np.sin((2*np.pi*(i+1))/T+fai))
            
#             elif season_type == "winter":
#                 T = 24
#                 fai = self.rng.choice([-np.pi/2,(5*np.pi)/12,np.pi/3])

#                 # coefficient
#                 beta_val = {}
#                 for i in range(12):
#                     beta_val[i+1] = A*np.abs(np.sin((2*np.pi*(i+1))/T+fai))
#             beta = [beta_val, season_type]

#         return beta
            
    
#     # determine median of event occurence time, which is the time at which the failure peobability becomes 50%.
#     def determine_median_time(self):
#         MEDIAN_TIME={}
#         for part_type in self.PARTS_DICT.keys():
#             flat_median = self.rng.integers(100, 365)
#             hard_median = int(flat_median/2)
#             MEDIAN_TIME[part_type] = {"FLAT":flat_median, "HARD":hard_median}
#         return MEDIAN_TIME

#     # uniformly determine k parameter for each part (uniform(kl,ku))
#     def uniform_determine_k_parameter(self):
#         K_PARAM={}
#         for part_type in self.PARTS_DICT.keys():
#             model_kind = self.PARTS_DICT[part_type]["failure_model"]
#             # Exponential model
#             if model_kind == "exponential":
#                 K_PARAM[part_type] = {"FLAT":0, "HARD":0}
#             # Weibull model
#             elif model_kind == "weibull":
#                 # FLAT
#                 k_flat = self.rng.random()*(3.0-1.0) + 1.0
#                 # HARD
#                 k_hard = self.rng.random()*(0.8-0.4) + 0.4
                
#                 K_PARAM[part_type] = {"FLAT":k_flat, "HARD":k_hard}

#             # Log-logistic model
#             elif model_kind == "log-logistic":
#                 # FLAT
#                 k_flat = self.rng.random()*(3.0-1.0) + 1.0
#                 # HARD
#                 k_hard = k_flat * self.rng.random()*(2.0-1.2) + 1.2

#                 K_PARAM[part_type] = {"FLAT":k_flat, "HARD":k_hard}

#             # Gompertz model
#             elif model_kind == "gompertz":
#                 # FLAT
#                 k_flat = self.rng.random()*(0.03-0.005) + 0.005
#                 # HARD
#                 k_hard = self.rng.random()*(0.1-0.03) + 0.03

#                 K_PARAM[part_type] = {"FLAT":k_flat, "HARD":k_hard}

#         return K_PARAM


#     # determine k parameter for each part (norm(const., 0.1))
#     def constant_determine_k_parameter(self):
#         K_PARAM={}
#         for part_type in self.PARTS_DICT.keys():
#             model_kind = self.PARTS_DICT[part_type]["failure_model"]
#             # Exponential model
#             if model_kind == "exponential":
#                 K_PARAM[part_type] = {"FLAT":0, "HARD":0}
#             # Weibull model
#             elif model_kind == "weibull":
#                 # FLAT
#                 kr_flat = 2.0
#                 v = self.rng.normal(np.log(kr_flat), 0.1)
#                 k_flat = np.exp(v)
#                 while k_flat <= 1:
#                     v = self.rng.normal(np.log(kr_flat), 0.1)
#                     k_flat = np.exp(v)

#                 # HARD
#                 kr_hard = 0.8
#                 v = self.rng.normal(np.log(kr_hard), 0.05)
#                 k_hard = np.exp(v)
#                 while k_hard <= 0 or k_hard > 1:
#                     v = self.rng.normal(np.log(kr_hard), 0.05)
#                     k_hard = np.exp(v)
                
#                 K_PARAM[part_type] = {"FLAT":k_flat, "HARD":k_hard}

#             # Log-logistic model
#             elif model_kind == "log-logistic":
#                 # FLAT
#                 kr_flat = 2.5
#                 v = self.rng.normal(np.log(kr_flat), 0.1)
#                 k_flat = np.exp(v)
#                 while k_flat <= 1:
#                     v = self.rng.normal(np.log(kr_flat), 0.1)
#                     k_flat = np.exp(v)
                
#                 # HARD
#                 k_hard = k_flat * 1.5

#                 K_PARAM[part_type] = {"FLAT":k_flat, "HARD":k_hard}

#             # Gompertz model
#             elif model_kind == "gompertz":
#                 # FLAT
#                 kr_flat = 0.4
#                 v = self.rng.normal(np.log(kr_flat), 0.1)
#                 k_flat = np.exp(v)
#                 while k_flat <= 0:
#                     v = self.rng.normal(np.log(kr_flat), 0.1)
#                     k_flat = np.exp(v)
                
#                 # HARD
#                 k_hard = k_flat * 1.5

#                 K_PARAM[part_type] = {"FLAT":k_flat, "HARD":k_hard}

#         return K_PARAM
    
#     # determine truck usage randomly
#     def determine_usage(self):
#         # flat:hard = 1:1
#         if self.rng.random()<0.5:
#             return "FLAT"
#         else:
#             return "HARD"
    
#     # manage holding trucks
#     def manage_trucks(self, time:int, delta_time:int, month:int):
#         events_trucks = []
#         for truck in self.fleet:
#             # evaluate part failure
#             evs = truck.checkup_parts(time=time, delta_time=delta_time, month=month)
#             events_trucks.extend(evs)

#             # truck is proceeding
#             truck.increment_age(delta_time=delta_time)
#         return events_trucks 