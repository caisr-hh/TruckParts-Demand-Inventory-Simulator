from typing import List, Dict, Tuple, Any, Optional, Iterable
import part as part_mod
import random

# part class
import importlib
importlib.reload(part_mod)
Part = part_mod.Part



# list of parts constituting one truck (part_type, count)
DEFAULT_PART_LIST: List[Tuple[str, int]] = [
    ("tire", 10),
    ("brake_pad", 2),
    ("oil_filter", 1),
    ("battery", 1)
]

# dictionary of failure model for default parts
DEFAULT_PART_FAILURE_MODEL = {
    "tire": {"kind": "exponential", "params": {"MTTF": 100}},
    "brake_pad": {"kind": "weibull", "params": {"lambda0": 1/200, "alpha0": 1.5}},
    "oil_filter": {"kind": "log-logstic", "params": {"lambda0": 1/200, "alpha0": 2.5}},
    "battery": {"kind": "gompertz", "params": {"lambda0": 1/100, "alpha0": 1/1000}}
}

# dictionary of available failure model
AVAILABLE_PART_MODEL = {
    "exponential": {"MTTF": 100},
    "weibull": {"lambda0": 1/200, "alpha0": 1.5},
    "log-logstic": {"lambda0": 1/200, "alpha0": 2.5},
    "gompertz": {"lambda0": 1/100, "alpha0": 1/1000}
}


# allocate default part id
def mk_default_part_id(truck_id: str, part_type: str, idx: int) -> str:
    return f"{truck_id}-{part_type}-{idx:03d}"

# allocate part id
def mk_part_id(truck_id: str, part_type: int) -> str:
    return f"{truck_id}-{part_type:03d}"


class Truck:
    def __init__(self, truck_id: str, model_id: str, n_parts: int=10,
                 part_setting: str = "RANDOM",
                 PART_LIST: Optional[Iterable[Tuple[str,int]]] = None,
                 FAILURE_MODEL: Optional[dict] = None):
        # identifier of this truck
        self.truck_id: str = truck_id
        self.model_id: str = model_id
        
        # elapsed time after the last replacement of either parts
        self.age: int = 0
        
        # parts constituting this truck
        self.parts: List[Part] = []
        if part_setting == "RANDOM":        # random setting 
            self.attach_random_part(n_parts=n_parts, FAILURE_MODEL=FAILURE_MODEL or AVAILABLE_PART_MODEL)
        elif part_setting == "DEFAULT":     # default setting from DEFAULT_PART_LIST
            self.attach_default_part(PART_LIST=list(PART_LIST) if PART_LIST is not None else DEFAULT_PART_LIST,
                FAILURE_MODEL=FAILURE_MODEL or DEFAULT_PART_FAILURE_MODEL)
    
    # attach parts to the truck
    def attach_random_part(self, n_parts: int, FAILURE_MODEL: dict):
        for i in range(n_parts):
            model_kind = random.choice(list(FAILURE_MODEL.keys()))

            # part identifier (type, id)
            part_type = i
            part_id = mk_part_id(self.truck_id, part_type)

            # attach part
            self.parts.append(Part(part_id=part_id, 
                                part_type=part_type, 
                                failure_model={"kind": model_kind, "params":FAILURE_MODEL[model_kind]}))


    # attach default parts to the truck
    def attach_default_part(self, PART_LIST: Iterable[Tuple[str,int]], 
                    FAILURE_MODEL: dict) -> None:
        for part_type, count in PART_LIST:
            failure_model = FAILURE_MODEL[part_type]
            for i in range(count):
                # allocate part id
                part_id = mk_default_part_id(self.truck_id, part_type, i)

                # attach part
                self.parts.append(Part(part_id=part_id, 
                                       part_type=part_type, failure_model=failure_model))


    # update operating conditions
    def update_conditions(self):
        pass
    
    # increment truck age and its parts ages
    def increment_age(self, delta_time: int):
        self.age += delta_time
        for part in self.parts:
            part.age += delta_time

    # daily checkup on each part 
    def checkup_parts(self, time: int, delta_time: int):
        events = []
        for part in self.parts:
            # print(time)
            ev = part.evaluate_failure(time=time, delta_time=delta_time, 
                                       truck_id=self.truck_id,
                                       model_id=self.model_id,
                                       truck_age=self.age)
            if ev:
                events.append(ev)
        return events
    

# create multiple trucks fleet
def create_fleet(n_trucks: int, model_id: str = "M0", 
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


    