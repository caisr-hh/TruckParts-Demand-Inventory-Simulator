from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    start_date: datetime
    total_date: int
    delta_date: int


class Simulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    


