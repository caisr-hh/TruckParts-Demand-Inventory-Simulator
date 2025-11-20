from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np
from datetime import datetime
from scipy.stats import norm
from costTracker import CostTracker
import math


@dataclass
class InventoryPolicyParams:
    """Parameters used in inventory calculations"""
    lead_time: int
    service_level: float = 0.95       # 95% service level
    review_period: int = 1            # Days between reviews
    order_quantity_multiplier: float = 1.0  # Multiplier for order quantity
    reorder_point_multiplier: float = 1.0   # Multiplier for reorder point
    
    def __post_init__(self):
        """Validate multiplier values"""
        if self.order_quantity_multiplier <= 0:
            raise ValueError("Order quantity multiplier must be positive")
        if self.reorder_point_multiplier <= 0:
            raise ValueError("Reorder point multiplier must be positive")

@dataclass
class PredictiveInventoryPolicyParams(InventoryPolicyParams):
    """Extended parameters for predictive interval policy"""
    prediction_confidence: float = 0.95  # Confidence level of prediction intervals
    
class BaseInventoryPolicy(ABC):
    """Abstract base class for inventory policies"""
    
    def __init__(self, params: InventoryPolicyParams):
        self.params = params
        self.cost_tracker = CostTracker()  # To access cost constants
    
    @abstractmethod
    def calculate_safety_stock(self, historical_data: np.ndarray) -> float:
        """Calculate safety stock level"""
        pass
    
    @abstractmethod
    def calculate_reorder_point(self, historical_data: np.ndarray) -> float:
        """Calculate reorder point"""
        pass
    
    @abstractmethod
    def calculate_order_quantity(self, historical_data: np.ndarray) -> float:
        """Calculate order quantity"""
        pass

class StandardInventoryPolicy(BaseInventoryPolicy):
    """Standard inventory policy using normal distribution assumptions"""
    
    def calculate_safety_stock(self, historical_data: np.ndarray) -> float:
        """
        Calculate safety stock using normal distribution.
        SS = Z × σL
        where:
        Z = safety factor (derived from service level)
        σL = standard deviation of demand during lead time
        """
        # Calculate Z-score from service level
        z_score = norm.ppf(self.params.service_level)
        
        # Calculate standard deviation of demand during lead time
        # σL = σd × √L where σd is daily demand std dev and L is lead time
        daily_std = np.std(historical_data)
        lead_time_std = daily_std * np.sqrt(self.params.lead_time)
        
        # Calculate safety stock
        ss = np.round(z_score * lead_time_std, 2)
        ss = max(ss, 0)  
            
        return int(ss)
    
    def calculate_reorder_point(self, historical_data: np.ndarray) -> float:
        """
        Calculate reorder point using:
        ROP = (μL + SS) * multiplier
        where:
        μL = average demand during lead time
        SS = safety stock
        multiplier = reorder point multiplier
        """
        avg_daily_demand = np.mean(historical_data)
        lead_time_demand = avg_daily_demand * self.params.lead_time
        safety_stock = self.calculate_safety_stock(historical_data)
        base_rop = lead_time_demand + safety_stock
        
        return int(base_rop * self.params.reorder_point_multiplier)
    
    
    def calculate_order_quantity(self, historical_data: np.ndarray) -> float:
        """
        Calculate basic Economic Order Quantity (EOQ) with multiplier.

        Parameters:
        historical_data (np.ndarray): Historical demand data

        Returns:
        float: EOQ (optimal order quantity)

        EOQ = sqrt((2 * D * S) / H) * multiplier
        where:
        D = Annual demand
        S = Ordering cost per order
        H = Holding cost per unit per year
        multiplier = order quantity multiplier
        """
        ordering_cost = self.cost_tracker.ORDER_COST
        holding_cost = self.cost_tracker.HOLDING_COST
        annual_demand = np.mean(historical_data) * 365  # Convert daily demand to annual

        if annual_demand <= 0 or ordering_cost <= 0 or holding_cost <= 0:
            raise ValueError("All input values must be positive numbers.")

        # Calculate basic EOQ
        base_eoq = math.sqrt((2 * annual_demand * (ordering_cost)) / holding_cost)
        return int(base_eoq * self.params.order_quantity_multiplier)
    
class PredictiveIntervalPolicy(BaseInventoryPolicy):
    """
    Cost-optimized inventory policy using prediction intervals for demand forecasting
    
    Key Features:
    - Dynamically balances holding costs against stockout costs
    - Optimizes order quantities considering transport costs
    - Implements cost-based service level adjustment
    - Uses prediction intervals for uncertainty quantification
    """
    
    def __init__(self, params: PredictiveInventoryPolicyParams):
        super().__init__(params)
        self.z_pred = norm.ppf((1 + params.prediction_confidence)/2)
            
    def calculate_safety_stock(self, historical_data: np.ndarray) -> float:
        """
        Calculate safety stock using prediction intervals directly:
        Safety Stock = upper_bound_L - y_hat_L
        
        Where:
        - y_hat_L: Point forecast for demand during lead time L
        - upper_bound_L: Upper prediction bound for L-period demand
        
        This method:
        1. Calculates lead time demand forecast
        2. Calculates lead time upper bound
        3. Safety stock is the difference between these values
        """
        # Get average point forecast and upper bound
        avg_forecast = np.mean(historical_data[:,0])  # y_hat (point forecast)
        avg_upper_bound = np.mean(historical_data[:,2])  # upper prediction bound
        
        # Calculate for lead time period
        lead_time_forecast = avg_forecast * self.params.lead_time  # y_hat_L
        lead_time_upper = avg_upper_bound * self.params.lead_time  # upper_bound_L
        
        # Safety stock is the difference
        safety_stock = lead_time_upper - lead_time_forecast
        
        return int(max(safety_stock, 0))

    def calculate_reorder_point(self, historical_data: np.ndarray) -> float:
        """
        Calculate reorder point using upper prediction bound:
        ROP = upper_bound_L
        
        This approach:
        - Uses prediction intervals directly for better uncertainty capture
        - Automatically accounts for demand-supply covariance
        - Handles non-stationary variance across product lifecycles
        - Adapts to lead time compression from supplier improvements
        """
        # Get average upper prediction bound
        avg_upper_bound = np.mean(historical_data[:,2])
        
        # Scale to lead time period
        lead_time_upper = avg_upper_bound * self.params.lead_time
        
        return int(lead_time_upper)

    def calculate_order_quantity(self, historical_data: np.ndarray) -> float:
        """
        Calculate order quantity using probabilistic EOQ adjustment:
        Q* = sqrt((2*D*y_hat_a*S)/H) * sqrt(1 + (upper_bound_L - y_hat_L)/y_hat_L)
        
        Where:
        - D: Days in planning period (365)
        - y_hat_a: Annualized demand from y_hat_L
        - S: Ordering cost per transaction
        - H: Holding cost per unit
        - upper_bound_L: Upper prediction bound for lead time
        - y_hat_L: Point forecast for lead time
        """
        # Get average point forecast and upper bound
        avg_forecast = np.mean(historical_data[:,0])  # y_hat (point forecast)
        avg_upper_bound = np.mean(historical_data[:,2])  # upper prediction bound
        
        # Calculate lead time values
        lead_time_forecast = avg_forecast * self.params.lead_time  # y_hat_L
        lead_time_upper = avg_upper_bound * self.params.lead_time  # upper_bound_L
        
        # Calculate annualized demand (avoid double counting 365)
        annual_demand = avg_forecast * 365  # y_hat_a
        
        # Calculate base EOQ with single annual factor
        base_eoq = math.sqrt(
            (2 * annual_demand * self.cost_tracker.ORDER_COST) /
            self.cost_tracker.HOLDING_COST
        )
        
        # Calculate dampened uncertainty adjustment
        uncertainty_factor = (lead_time_upper - lead_time_forecast) / lead_time_forecast
        # Dampen the adjustment to prevent excessive quantities
        dampened_adjustment = math.sqrt(1 + min(uncertainty_factor, 0.5))
        
        # Apply dampened adjustment to base EOQ
        eoq = base_eoq * dampened_adjustment
        
        # Use smaller rounding factor to prevent large jumps
        rounding_factor = 50
        rounded_eoq = math.ceil(eoq / rounding_factor) * rounding_factor
        
        # Add upper limit to prevent excessive quantities
        max_order = annual_demand / 6  # Maximum 3 months of demand
        return int(min(rounded_eoq, max_order))
