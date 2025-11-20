class Timestamp:
    """Represents a point in time with days and hours"""
    def __init__(self, days: int, hours: float):
        self.days = days
        self.hours = hours
    
    def __str__(self) -> str:
        return f"Day {self.days}, {self.hours:.1f}h"
    
    @classmethod
    def from_hours(cls, total_hours: float) -> 'Timestamp':
        days = int(total_hours // 24)
        hours = total_hours % 24
        return cls(days, hours)
    
    def to_hours(self) -> float:
        """Convert timestamp to total hours for calculations"""
        return (self.days * 24) + self.hours
    
    def __add__(self, hours: float) -> 'Timestamp':
        """Add hours to timestamp"""
        total_hours = self.to_hours() + hours
        return Timestamp.from_hours(total_hours)