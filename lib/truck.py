from typing import List, Dict, Tuple, Any, Optional, Iterable
from part import Part

# list of parts constituting one truck (part_type, count)
DEFAULT_PART_LIST: List[Tuple[str, int]] = [
    ("tire", 10),
    ("brake_pad", 2),
    ("oil_filter", 1),
    ("battery", 1)
]

# dictionary of MTTF (average lifetime) of each part
DEFAULT_MTTF_DAYS: Dict[str, float] = {
    "tire": 300,
    "brake_pad": 200,
    "oil_filter": 50,
    "battery": 350
}

# allocate part id
def mk_part_id(truck_id: str, part_type: str, idx: int) -> str:
    return f"{truck_id}-{part_type}-{idx:03d}"


class Truck:
    def __init__(self, truck_id: str, model_id: str, 
                 auto_part_setting: bool = True,
                 PART_LIST: Optional[Iterable[Tuple[str,int]]] = None,
                 MTTF_DAYS: Optional[Dict[str, float]] = None ):
        # identifier of this truck
        self.truck_id: str = truck_id
        self.model_id: str = model_id
        
        # elapsed time after the last replacement of either parts
        self.age: int = 0
        
        # parts constituting this truck
        self.parts: List[Part] = []
        if auto_part_setting:
            self.attach_part(PART_LIST=list(PART_LIST) if PART_LIST is not None else DEFAULT_PART_LIST,
                MTTF=MTTF_DAYS or DEFAULT_MTTF_DAYS)
    
    # attach part to this truck
    def attach_part(self, PART_LIST: Iterable[Tuple[str,int]], 
                    MTTF: Dict[str, float]) -> None:
        for part_type, count in PART_LIST:
            mttf = MTTF[part_type]  # MTTF of the part
            for i in range(count):
                # allocate part id
                part_id = mk_part_id(self.truck_id, part_type, i)

                # attach part
                self.parts.append(Part(part_id=part_id, part_type=part_type, mttf=mttf))


    # update operating conditions
    def update_conditions(self):
        pass

    # daily checkup on each part 
    def checkup_parts(self):
        pass
    

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


    