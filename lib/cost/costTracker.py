from orderManagement import OrderPriority

class CostTracker:
    def __init__(self):
        # Cost constants
        self.HOLDING_COST_RATE = 0.15 
        self.PART_VALUE = 0.13
        self.ORDER_COST = 100  # Fixed cost per order (not per quantity)
        self.RUSH_ORDER_COST = 165  # Fixed cost per urgent order (not per quantity)
        self.TRANSPORT_COST_PER_KG = 2
        self.RETURN_COST_PER_KG = 2
        self.BADWILL_COST = 1000
        self.KG_PER_PART = 0.001
        self.HOLDING_COST = self.HOLDING_COST_RATE * self.PART_VALUE
        self.vital_code = 3
        self.TARGET_SERVICE_LEVEL = 0.95

        
        # Initialize cost trackers as variables
        self.total_holding_cost = 0
        self.total_order_cost = 0
        self.total_transport_cost = 0
        self.total_return_cost = 0
        self.total_badwill_cost = 0
        
        # Track number of orders
        self.number_of_orders = 0
        self.number_of_urgent_orders = 0
    
    def calculate_holding_cost(self, inventory_level):
        """Calculate holding cost directly from inventory level"""
        daily_holding_rate = self.HOLDING_COST_RATE * self.PART_VALUE
        cost = (daily_holding_rate * inventory_level)/365
        self.total_holding_cost += cost
        return cost

    def calculate_order_and_transport_costs(self, quantity, priority: OrderPriority, return_quantity=0):
        """
        Calculate combined order processing and transport costs
        
        Order cost is tracked by incrementing order counters
        Transport cost is calculated based on quantity
        """
        total_kg = quantity * self.KG_PER_PART
        return_kg = return_quantity * self.KG_PER_PART
        
        # Track the number of orders by priority
        if priority == OrderPriority.URGENT:
            self.number_of_urgent_orders += 1
        else:
            self.number_of_orders += 1
            
        # Calculate transport and return costs (these vary with quantity)
        transport_cost = self.TRANSPORT_COST_PER_KG * total_kg
        return_cost = self.RETURN_COST_PER_KG * return_kg
        
        # Update transport and return cost totals
        self.total_transport_cost += transport_cost
        self.total_return_cost += return_cost
        
        # Calculate the current total order cost based on order counts
        # This updates the total_order_cost variable directly
        self.total_order_cost = (self.number_of_orders * self.ORDER_COST) + \
                               (self.number_of_urgent_orders * self.RUSH_ORDER_COST)
        
        # Return the combined cost for this specific order
        # For the individual order, we still need to apply the appropriate base cost
        base_cost = self.RUSH_ORDER_COST if priority == OrderPriority.URGENT else self.ORDER_COST
        return base_cost + transport_cost + return_cost

    def get_total_costs(self):
        """Get total of all costs and individual breakdowns"""
        # Calculate badwill cost based on the formula
        # Only if vital code is not 3 or 4
        if self.vital_code in [1,2,3, 4]:
            total_order_lines = self.number_of_orders + self.number_of_urgent_orders
            self.total_badwill_cost = (1 - self.TARGET_SERVICE_LEVEL) * total_order_lines * self.BADWILL_COST
        else:
            self.total_badwill_cost = 0
        
        # Calculate regular and rush order costs separately
        regular_order_cost = self.number_of_orders * self.ORDER_COST
        rush_order_cost = self.number_of_urgent_orders * self.RUSH_ORDER_COST
                
        # Calculate total
        total = (self.total_holding_cost +
                self.total_order_cost +
                self.total_transport_cost +
                self.total_return_cost +
                self.total_badwill_cost)
                
        return {
            'holding_cost': self.total_holding_cost,
            'order_cost': self.total_order_cost,
            'regular_order_cost': regular_order_cost,
            'rush_order_cost': rush_order_cost,
            'transport_cost': self.total_transport_cost,
            'return_cost': self.total_return_cost,
            'badwill_cost': self.total_badwill_cost,
            'total_cost': total
        }

    def get_costs(self):
        return {
        "HOLDING_COST_RATE": self.HOLDING_COST_RATE,
        "PART_VALUE": self.PART_VALUE,
        "ORDER_COST": self.ORDER_COST,
        "RUSH_ORDER_COST": self.RUSH_ORDER_COST,
        "TRANSPORT_COST_PER_KG": self.TRANSPORT_COST_PER_KG,
        "RETURN_COST_PER_KG": self.RETURN_COST_PER_KG,
        "BADWILL_COST": self.BADWILL_COST,
        "KG_PER_PART": self.KG_PER_PART,
        "HOLDING_COST": self.HOLDING_COST
         }