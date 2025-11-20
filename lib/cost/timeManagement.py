from datetime import datetime, timedelta
from typing import Optional

class SimulationClock:
    """
    Manages simulation time without business hour restrictions.
    Matches the time handling approach used in OrderManagement.
    """
    def __init__(self, start_time: datetime):
        self.current_time = start_time
        self.total_steps = 0
        
    def advance_time(self, hours: float = 1.0) -> datetime:
        """
        Advance the simulation clock by the specified number of hours.
        Supports fractional hours.
        
        Args:
            hours: Number of hours to advance (can be fractional)
        
        Returns:
            Updated current time
        """
        self.current_time += timedelta(hours=hours)
        self.total_steps += 1
        return self.current_time
    
    def set_time(self, new_time: datetime) -> None:
        """
        Set the simulation clock to a specific time.
        
        Args:
            new_time: Time to set the clock to
        """
        if new_time < self.current_time:
            raise ValueError("Cannot set time to a past moment")
        self.current_time = new_time
    
    def get_time_until(self, target_time: datetime) -> float:
        """
        Calculate the number of hours between current time and target time.
        
        Args:
            target_time: Target time to calculate difference to
            
        Returns:
            Number of hours between current time and target time
        """
        if target_time < self.current_time:
            raise ValueError("Target time cannot be earlier than current time")
        
        time_diff = target_time - self.current_time
        return time_diff.total_seconds() / 3600  # Convert to hours
    
    def get_current_time(self) -> datetime:
        """Get current simulation time."""
        return self.current_time
    
    def get_total_steps(self) -> int:
        """Get total number of simulation steps taken."""
        return self.total_steps
    
    def format_time(self) -> str:
        """Format current time as string, matching OrderManager format."""
        return self.current_time.strftime("%Y-%m-%d %H:%M")
    
    def __str__(self) -> str:
        """String representation of current simulation state."""
        return f"SimulationClock(time={self.format_time()}, steps={self.total_steps})"