from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from orderManagement import OrderManager, Order, OrderPriority
from costTracker import CostTracker

@dataclass
class InventoryState:
    """Tracks inventory levels and related metrics for a specific part at a location"""
    spare_part_id: str
    location_id: int
    current_stock: float = 0
    reorder_point: float = 0
    backorders: float = 0
    lost_sales: int = 0
    total_demand: float = 0          # Track total demand
    fulfilled_demand: float = 0       # Track total fulfilled demand (immediate + backorder)
    immediate_fulfilled: float = 0    # Track only immediately fulfilled demand
    
    # Historical tracking
    stock_history: List[tuple[datetime, float]] = field(default_factory=list)
    backorder_history: List[tuple[datetime, float]] = field(default_factory=list)
    lost_sales_history: List[tuple[datetime, int]] = field(default_factory=list)

class StateManager:
    """
    Manages the complete state of the simulation, including inventory levels,
    order processing, and performance metrics.
    """
    def __init__(self, 
                order_manager: OrderManager,
                cost_tracker: CostTracker):
        self.order_manager = order_manager
        self.cost_tracker = cost_tracker
        
        # Dictionary to store inventory state for each part at each location
        # Key: (spare_part_id, location_id)
        self.inventory_states: Dict[tuple[str, int], InventoryState] = {}
        
        # Track simulation KPIs
        self.service_level_history: List[tuple[datetime, float]] = []
        self.cost_history: List[tuple[datetime, Dict]] = []
        
    def initialize_inventory(self, 
                         spare_part_id: str,
                         location_id: int,
                         initial_stock: float,
                         reorder_point: float) -> None:
        """
        Initialize inventory state for a specific part at a location.
        """
        state = InventoryState(
            spare_part_id=spare_part_id,
            location_id=location_id,
            current_stock=initial_stock,
            reorder_point=reorder_point
        )
        self.inventory_states[(spare_part_id, location_id)] = state
        
    def process_demand(self,
                    spare_part_id: str,
                    location_id: int,
                    quantity: float,
                    current_time: datetime) -> float:
        """
        Process a demand request and update inventory state.
        Returns the unfulfilled quantity that needs to be backordered.
        Also tracks stockouts when demand cannot be fulfilled from current stock.
        """
        state = self.inventory_states.get((spare_part_id, location_id))
        if not state:
            raise ValueError(f"No inventory state for part {spare_part_id} at location {location_id}")
            
        # Update total demand
        state.total_demand += quantity
        
        # Try to fulfill from stock
        fulfilled_quantity = min(quantity, state.current_stock)
        state.current_stock -= fulfilled_quantity
        state.fulfilled_demand += fulfilled_quantity
        
        # Track immediate fulfillment separately
        state.immediate_fulfilled += fulfilled_quantity
        
        # Calculate unfulfilled quantity and track stockout
        unfulfilled = quantity - fulfilled_quantity
        if unfulfilled > 0:
            state.backorders += unfulfilled
            # Track stockout event in lost_sales_history (one stockout per unfulfilled demand)
            state.lost_sales += 1
            state.lost_sales_history.append((current_time, state.lost_sales))
        # Update histories
        state.stock_history.append((current_time, state.current_stock))
        if state.backorders > 0:
            state.backorder_history.append((current_time, state.backorders))
            
        return unfulfilled
            
    def process_delivery(self,
                      order: Order,
                      current_time: datetime) -> None:
        """
        Process an order delivery and update inventory state.
        """
        state = self.inventory_states.get((order.spare_part_id, order.location_id))
        if not state:
            raise ValueError(f"No inventory state for part {order.spare_part_id} at location {order.location_id}")
            
        # Add delivered quantity to stock
        state.current_stock += order.quantity
        
        # Update stock history with new delivery
        state.stock_history.append((current_time, state.current_stock))
        
        # Then fulfill any backorders if possible
        if state.backorders > 0:
            fulfilled_backorders = min(state.backorders, state.current_stock)
            state.current_stock -= fulfilled_backorders
            state.backorders -= fulfilled_backorders
            # Update total fulfilled demand but not immediate_fulfilled since these are backorders
            state.fulfilled_demand += fulfilled_backorders
            
            # Update histories after fulfilling backorders
            state.stock_history.append((current_time, state.current_stock))
            if state.backorders > 0:
                state.backorder_history.append((current_time, state.backorders))
                
    def update_service_level(self, current_time: datetime) -> float:
        """
        Calculate and update current service level (Immediate Fill Rate).
        Fill Rate = Immediately Fulfilled Demand / Total Demand
        """
        total_demand = 0
        total_fulfilled = 0
        
        for state in self.inventory_states.values():
            total_demand += state.total_demand
            total_fulfilled += state.immediate_fulfilled
            
        service_level = total_fulfilled / total_demand if total_demand > 0 else 1.0
        self.service_level_history.append((current_time, service_level))
        return service_level
    
    def update_costs(self, current_time: datetime) -> None:
        """
        Update cost tracking.
        """
        # Calculate holding costs for all inventory
        for state in self.inventory_states.values():
            self.cost_tracker.calculate_holding_cost(state.current_stock)
            
        # Get total costs and add to history
        current_costs = self.cost_tracker.get_total_costs()
        self.cost_history.append((current_time, current_costs))
        
    def get_current_state(self, spare_part_id: str, location_id: int) -> Optional[InventoryState]:
        """
        Get current inventory state for a specific part at a location.
        """
        return self.inventory_states.get((spare_part_id, location_id))
    
    def get_all_states(self) -> Dict[tuple[str, int], InventoryState]:
        """
        Get all current inventory states.
        """
        return self.inventory_states.copy()
