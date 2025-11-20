from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional, List
from queue import PriorityQueue

class EventType(Enum):
    DEMAND = auto()          # New demand arrives
    DELIVERY = auto()        # Order delivery
    INVENTORY_CHECK = auto() # Check stock levels
    REORDER = auto()         # Place new order

@dataclass
class Event:
    """
    Represents a simulation event.
    """
    time: datetime
    event_type: EventType
    data: Any = None

    def __eq__(self, other):
        if not isinstance(other, Event):
            return NotImplemented
        return self.time == other.time
    
    def __lt__(self, other):
        if not isinstance(other, Event):
            return NotImplemented
        return self.time < other.time

class EventQueue:
    """
    Manages the queue of simulation events.
    """
    def __init__(self):
        self.queue = PriorityQueue()
        self._event_count = 0  # Track total events for logging/debugging
    
    def add_event(self, 
                 time: datetime,
                 event_type: EventType,
                 data: Any = None) -> None:
        """
        Add a new event to the queue.
        
        Args:
            time: When the event should occur
            event_type: Type of event
            data: Associated event data
        """
        event = Event(time, event_type, data)
        self.queue.put(event)
        self._event_count += 1
    
    def get_next_event(self) -> Optional[Event]:
        """
        Get the next event from the queue.
        Returns None if queue is empty.
        """
        if self.queue.empty():
            return None
        return self.queue.get()
    
    def peek_next_event(self) -> Optional[Event]:
        """
        Look at the next event without removing it.
        Returns None if queue is empty.
        """
        if self.queue.empty():
            return None
        event = self.queue.get()
        self.queue.put(event)
        return event
    
    def get_events_until(self, time: datetime) -> List[Event]:
        """
        Get all events scheduled to occur up to the specified time.
        
        Args:
            time: Get events scheduled up to this time
            
        Returns:
            List of events ordered by time and priority
        """
        events = []
        while not self.queue.empty():
            event = self.peek_next_event()
            if event and event.time <= time:
                events.append(self.get_next_event())
            else:
                break
        return events
    
    def clear(self) -> None:
        """
        Clear all events from the queue.
        """
        while not self.queue.empty():
            self.queue.get()
        self._event_count = 0
    
    def get_queue_size(self) -> int:
        """
        Get current number of events in queue.
        """
        return self.queue.qsize()
    
    def get_total_events_processed(self) -> int:
        """
        Get total number of events that have been added to queue.
        """
        return self._event_count
