from dataclasses import dataclass

# Demand data
@dataclass (frozen=True)    
class DemandEvent:
    time: int
    truck_id: str
    model_id: str
    truck_age: int
    part_id: str
    part_type: str
    part_age: int

# Failure data at each iteration
@dataclass (frozen=True)
class FailureData:
    time: int
    dealer_id: str
    truck_id: str
    model_id: str
    truck_age: int
    part_id: str
    part_type: str
    part_age: int
    failure: int