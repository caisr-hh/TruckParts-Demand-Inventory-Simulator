from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class OrderStatus(Enum):
    PROCESSING = "processing"
    DELIVERED = "delivered"

class OrderPriority(Enum):
    URGENT = "urgent"
    NON_URGENT = "non_urgent"

@dataclass
class Order:
    order_id: str
    spare_part_id: str
    location_id: int
    quantity: float
    priority: OrderPriority
    created_at: datetime
    lead_time: int      # lead time in hours
    status: OrderStatus = OrderStatus.PROCESSING
    delivered_at: Optional[datetime] = None
    
    @property
    def delivery_due_time(self) -> datetime:
        """Calculate when the order should be delivered based on lead time."""
        return self.created_at + timedelta(hours=self.lead_time)
    
    @property
    def is_delivered(self) -> bool:
        """Check if order has been delivered."""
        return self.status == OrderStatus.DELIVERED

class OrderManager:
    def __init__(self, urgent_lead_time: int = 12, non_urgent_lead_time: int = 24):
        """
        Initialize OrderManager with default lead times in hours.
        
        Args:
            urgent_lead_time: Default lead time for urgent orders (hours)
            non_urgent_lead_time: Default lead time for non-urgent orders (hours)
        """
        self.orders: Dict[str, Order] = {}
        self._order_counter = 0
        self.urgent_lead_time = urgent_lead_time
        self.non_urgent_lead_time = non_urgent_lead_time
    
    def _generate_order_id(self) -> str:
        self._order_counter += 1
        return f"ORD{self._order_counter:06d}"
    
    def _get_default_lead_time(self, priority: OrderPriority) -> int:
        return (
            self.urgent_lead_time 
            if priority == OrderPriority.URGENT 
            else self.non_urgent_lead_time
        )
    
    def create_order(
        self,
        spare_part_id: str,
        location_id: int,
        quantity: float,
        current_time: datetime,
        priority: OrderPriority = OrderPriority.NON_URGENT,
        lead_time: Optional[int] = None
    ) -> Order:
        order_id = self._generate_order_id()
        actual_lead_time = lead_time if lead_time is not None else self._get_default_lead_time(priority)
        
        order = Order(
            order_id=order_id,
            spare_part_id=spare_part_id,
            location_id=location_id,
            quantity=quantity,
            priority=priority,
            created_at=current_time,
            lead_time=actual_lead_time
        )
        
        self.orders[order_id] = order
        return order
    
    def deliver_order(self, order_id: str, current_time: datetime) -> None:
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = OrderStatus.DELIVERED
            order.delivered_at = current_time
    
    def get_pending_deliveries(self, current_time: datetime) -> List[Order]:
        """
        Get list of orders that are still processing and have not reached their due time.
        Returns urgent orders first, then non-urgent orders.
        
        Args:
            current_time: Current simulation time
        
        Returns:
            List of pending orders, sorted by priority
        """
        pending = [
            order for order in self.orders.values()
            if order.status == OrderStatus.PROCESSING and current_time < order.delivery_due_time
        ]
        
        return sorted(
            pending,
            key=lambda x: (
                0 if x.priority == OrderPriority.URGENT else 1,
                x.delivery_due_time
            )
        )
    
    def auto_deliver_due_orders(self, current_time: datetime) -> None:
        """
        Automatically deliver orders that have reached their due time.
        
        Args:
            current_time: Current simulation time
        """
        for order in self.orders.values():
            if (order.status == OrderStatus.PROCESSING and 
                current_time >= order.delivery_due_time):
                self.deliver_order(order.order_id, current_time)
    
    @staticmethod
    def format_time(dt: datetime) -> str:
        """Format datetime to string."""
        return dt.strftime("%Y-%m-%d %H:%M")