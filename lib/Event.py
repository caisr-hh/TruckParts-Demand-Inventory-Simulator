from dataclasses import dataclass

@dataclass (frozen=True)    # 
class DemandEvent:
    time: int
    truck_id: str
    model_id: str
    truck_age: int
    part_id: str
    part_type: str
    part_age: int