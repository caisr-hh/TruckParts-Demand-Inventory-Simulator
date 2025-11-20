import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

class MetricsPlotter:
    """Class for plotting and comparing metrics between different simulation runs"""
    
    def __init__(self, figsize=(12, 6)):
        """
        Initialize the plotter with default figure size
        
        Args:
            figsize (tuple): Default figure size for plots
        """
        self.figsize = figsize
        # Set style
        plt.style.use('ggplot')  # Using a built-in matplotlib style
    
    def plot_stock_levels(self, sim1_results: Dict[str, Any], sim2_results: Dict[str, Any], 
                         labels: tuple = ("Actual Demand", "Predicted Demand")):
        """
        Plot stock levels over time for two simulations
        
        Args:
            sim1_results: First simulation results
            sim2_results: Second simulation results
            labels: Tuple of labels for the two simulations
        """
        plt.figure(figsize=self.figsize)
        
        # Extract timestamps and stock levels from history
        dates1 = [ts for ts, _ in sim1_results['stock_history']]
        dates2 = [ts for ts, _ in sim2_results['stock_history']]
        stock_levels1 = [level for _, level in sim1_results['stock_history']]
        stock_levels2 = [level for _, level in sim2_results['stock_history']]
        
        plt.plot(dates1, stock_levels1, label=labels[0], linewidth=2)
        plt.plot(dates2, stock_levels2, label=labels[1], linewidth=2)
        
        plt.title("Stock Level Comparison", fontsize=12, pad=20)
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("Stock Level", fontsize=10)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    def plot_cost_comparison(self, sim1_results: Dict[str, Any], sim2_results: Dict[str, Any],
                           labels: tuple = ("Actual Demand", "Predicted Demand")):
        """
        Plot cost comparison between two simulations with enhanced visualization
        
        Args:
            sim1_results: First simulation results
            sim2_results: Second simulation results
            labels: Tuple of labels for the two simulations
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Get total costs from detailed costs
        total_costs1 = sim1_results['detailed_costs']['total_cost']
        total_costs2 = sim2_results['detailed_costs']['total_cost']
        
        # Plot total costs with improved formatting
        ax1.bar([0, 1], [total_costs1, total_costs2], color=['#2ecc71', '#3498db'])
        ax1.set_title("Total Cost Comparison", fontsize=12, pad=20)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.set_ylabel("Total Cost (SEK)", fontsize=10)
        
        # Add value labels on bars
        for i, cost in enumerate([total_costs1, total_costs2]):
            ax1.text(i, cost, f'{cost:,.2f} SEK', ha='center', va='bottom')
        
        ax1.grid(True, alpha=0.3)
        
        # Enhanced cost breakdown
        # Check if we have the separated costs
        has_separated_costs = (
            'regular_order_cost' in sim1_results['detailed_costs'] and
            'rush_order_cost' in sim1_results['detailed_costs'] and
            'regular_order_cost' in sim2_results['detailed_costs'] and
            'rush_order_cost' in sim2_results['detailed_costs']
        )
        
        if has_separated_costs:
            # Use separated order costs
            cost_types = ['holding_cost', 'regular_order_cost', 'rush_order_cost', 'transport_cost', 
                        'return_cost', 'badwill_cost']
            cost_labels = ['Holding Cost', 'Regular Order Cost', 'Rush Order Cost', 'Transport Cost',
                          'Return Cost', 'Badwill Cost']
            colors = ['#1abc9c', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#f1c40f']
        else:
            # Use original cost categories
            cost_types = ['holding_cost', 'order_cost', 'transport_cost', 
                         'return_cost', 'badwill_cost']
            cost_labels = ['Holding Cost', 'Order Cost', 'Transport Cost', 
                          'Return Cost', 'Badwill Cost']
            colors = ['#1abc9c', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f']
        
        x = np.arange(2)
        width = 0.15
        
        # Calculate total for percentage
        total1 = sum(sim1_results['detailed_costs'][ct] for ct in cost_types)
        total2 = sum(sim2_results['detailed_costs'][ct] for ct in cost_types)
        
        for i, (cost_type, label, color) in enumerate(zip(cost_types, cost_labels, colors)):
            costs = [sim1_results['detailed_costs'][cost_type], 
                    sim2_results['detailed_costs'][cost_type]]
            bars = ax2.bar(x + i*width, costs, width, 
                          label=label,
                          color=color)
            
            # Add percentage labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                percentage = (height / (total1 if j == 0 else total2)) * 100
                ax2.text(bar.get_x() + bar.get_width()/2, height,
                        f'{percentage:.1f}%', ha='center', va='bottom',
                        rotation=90)
        
        ax2.set_title("Cost Breakdown by Category", fontsize=12, pad=20)
        ax2.set_xticks(x + width*len(cost_types)/2 - width/2)
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.set_ylabel("Cost (SEK)", fontsize=10)
        ax2.legend(loc='upper right', framealpha=0.9, bbox_to_anchor=(1.15, 1))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def plot_service_metrics(self, sim1_results: Dict[str, Any], sim2_results: Dict[str, Any],
                           labels: tuple = ("Actual Demand", "Predicted Demand")):
        """
        Plot service level metrics comparison
        
        Args:
            sim1_results: First simulation results
            sim2_results: Second simulation results
            labels: Tuple of labels for the two simulations
        """
        plt.figure(figsize=self.figsize)
        
        # Extract timestamps and service levels
        dates1, levels1 = zip(*sim1_results['service_level_history'])
        dates2, levels2 = zip(*sim2_results['service_level_history'])
        
        # Calculate cumulative service levels
        cum_service_level1 = np.cumsum(levels1) / np.arange(1, len(levels1) + 1)
        cum_service_level2 = np.cumsum(levels2) / np.arange(1, len(levels2) + 1)
        
        # Convert to percentage
        cum_service_level1 = cum_service_level1 * 100
        cum_service_level2 = cum_service_level2 * 100
        
        # Create figure with improved style
        plt.figure(figsize=self.figsize)
        plt.plot(dates1, cum_service_level1, label=labels[0], linewidth=2)
        plt.plot(dates2, cum_service_level2, label=labels[1], linewidth=2)
        
        # Add horizontal target line at 95%
        plt.axhline(y=95, color='r', linestyle='--', alpha=0.5, label='Target (95%)')
        
        plt.title("Cumulative Service Level Over Time", fontsize=12, pad=20)
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("Service Level (%)", fontsize=10)
        plt.legend(loc='lower right', framealpha=0.9)
        plt.grid(True, alpha=0.3)
        
        # Set y-axis limits with some padding
        plt.ylim(0, 105)
        plt.tight_layout()
    
    def plot_order_patterns(self, sim1_results: Dict[str, Any], sim2_results: Dict[str, Any],
                          labels: tuple = ("Actual Demand", "Predicted Demand")):
        """
        Plot order patterns comparison
        
        Args:
            sim1_results: First simulation results
            sim2_results: Second simulation results
            labels: Tuple of labels for the two simulations
        """
        plt.figure(figsize=self.figsize)
        
        # Extract dates and stock levels
        dates1 = [ts for ts, _ in sim1_results['stock_history']]
        dates2 = [ts for ts, _ in sim2_results['stock_history']]
        stock_levels1 = [level for _, level in sim1_results['stock_history']]
        stock_levels2 = [level for _, level in sim2_results['stock_history']]
        
        # Calculate changes in stock levels as approximation of orders
        order_approx1 = [j-i for i, j in zip(stock_levels1[:-1], stock_levels1[1:])]
        order_approx2 = [j-i for i, j in zip(stock_levels2[:-1], stock_levels2[1:])]
        
        # Plot only positive changes (orders)
        orders1 = [(d, val) for d, val in zip(dates1[1:], order_approx1) if val > 0]
        orders2 = [(d, val) for d, val in zip(dates2[1:], order_approx2) if val > 0]
        
        if orders1:
            x1, y1 = zip(*orders1)
            plt.scatter(x1, y1, label=f"{labels[0]} Orders", alpha=0.7, s=100)
        if orders2:
            x2, y2 = zip(*orders2)
            plt.scatter(x2, y2, label=f"{labels[1]} Orders", alpha=0.7, s=100)
        
        plt.title("Order Quantities Over Time", fontsize=12, pad=20)
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("Order Quantity", fontsize=10)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    def plot_forecast_vs_actual(self, actual_demand: np.ndarray, forecast_demand: np.ndarray, 
                              title: str = "Forecast vs Actual Demand"):
        """
        Plot forecast demand against actual demand over time
        
        Args:
            actual_demand: Array of actual demand values
            forecast_demand: Array of forecasted demand values
            title: Title for the plot (optional)
        """
        plt.figure(figsize=self.figsize)
        
        # Plot both lines using index for x-axis
        plt.plot(actual_demand, label='Actual Demand', linewidth=2, color='#0000FF')
        plt.plot(forecast_demand, label='Forecast Demand', linewidth=2, color='#FF0000', linestyle='--')
        
        # Calculate and display MAPE (excluding zero values in actual demand)
        non_zero_mask = actual_demand != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actual_demand[non_zero_mask] - forecast_demand[non_zero_mask]) / 
                                actual_demand[non_zero_mask])) * 100
            plt.text(0.02, 0.98, f'MAPE: {mape:.2f}%', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                verticalalignment='top')
        
        plt.title(title, fontsize=12, pad=20)
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("Demand", fontsize=10)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
    def plot_demand_vs_stock(self, sim1_results: Dict[str, Any], sim2_results: Dict[str, Any],
                           actual_demand: np.ndarray, labels: tuple = ("Actual Demand", "Predicted Demand")):
        """
        Plot demand versus stock levels
        
        Args:
            sim1_results: First simulation results
            sim2_results: Second simulation results
            actual_demand: Array of actual demand values
            labels: Tuple of labels for the two simulations
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Extract dates
        dates1 = [ts for ts, _ in sim1_results['stock_history']]
        dates2 = [ts for ts, _ in sim2_results['stock_history']]
        
        # First simulation
        stock_levels1 = [level for _, level in sim1_results['stock_history']]
        min_len1 = min(len(dates1), len(actual_demand))
        ax1.plot(dates1[:min_len1], stock_levels1[:min_len1], label='Stock Level', linewidth=2)
        ax1.plot(dates1[:min_len1], actual_demand[:min_len1], label='Actual Demand', alpha=0.7, linewidth=2, linestyle='--')
        ax1.set_title(f"Demand vs Stock Level - {labels[0]}", fontsize=12, pad=20)
        ax1.set_xlabel("Date", fontsize=10)
        ax1.set_ylabel("Quantity", fontsize=10)
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Second simulation
        stock_levels2 = [level for _, level in sim2_results['stock_history']]
        min_len2 = min(len(dates2), len(actual_demand))
        ax2.plot(dates2[:min_len2], stock_levels2[:min_len2], label='Stock Level', linewidth=2)
        ax2.plot(dates2[:min_len2], actual_demand[:min_len2], label='Actual Demand', alpha=0.7, linewidth=2, linestyle='--')
        ax2.set_title(f"Demand vs Stock Level - {labels[1]}", fontsize=12, pad=20)
        ax2.set_xlabel("Date", fontsize=10)
        ax2.set_ylabel("Quantity", fontsize=10)
        ax2.legend(loc='upper right', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()