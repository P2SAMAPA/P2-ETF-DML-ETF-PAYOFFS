"""
execution_models.py  —  Execution Cost Models with Analytical Derivatives
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class AlmgrenChrissModel:
    """
    Almgren-Chriss optimal execution model with analytical derivatives.
    """
    
    def __init__(self, temp_impact: float = 0.1, perm_impact: float = 0.01,
                 volatility: float = 0.2, risk_aversion: float = 0.5):
        self.temp_impact = temp_impact
        self.perm_impact = perm_impact
        self.volatility = volatility
        self.risk_aversion = risk_aversion
        
    def execution_cost(self, order_size: float, volume: float, 
                       time_horizon: int = 5) -> float:
        """
        Calculate Almgren-Chriss execution cost.
        """
        # Daily volume
        daily_volume = volume / time_horizon
        
        # Temporary impact
        temp_cost = self.temp_impact * (order_size / daily_volume) ** 0.5
        
        # Permanent impact
        perm_cost = self.perm_impact * (order_size / daily_volume)
        
        # Risk cost
        risk_cost = 0.5 * self.risk_aversion * self.volatility ** 2 * time_horizon
        
        return temp_cost + perm_cost + risk_cost
    
    def analytical_gradient(self, order_size: float, volume: float,
                           time_horizon: int = 5) -> Dict:
        """
        Analytical gradient of execution cost.
        """
        daily_volume = volume / time_horizon
        
        # Gradient of temporary impact
        d_temp = self.temp_impact * 0.5 * (order_size / daily_volume) ** (-0.5) * (1 / daily_volume)
        
        # Gradient of permanent impact
        d_perm = self.perm_impact * (1 / daily_volume)
        
        # Gradient of risk cost is zero with respect to order_size
        d_risk = 0.0
        
        return {
            "d_temp": d_temp,
            "d_perm": d_perm,
            "d_risk": d_risk,
            "d_total": d_temp + d_perm + d_risk
        }


class SquareRootImpactModel:
    """
    Square-root price impact model.
    """
    
    def __init__(self, coefficient: float = 0.1, gamma: float = 0.5):
        self.coefficient = coefficient
        self.gamma = gamma
        
    def impact(self, order_size: float, volume: float) -> float:
        """Calculate price impact."""
        return self.coefficient * (order_size / volume) ** self.gamma
    
    def analytical_gradient(self, order_size: float, volume: float) -> float:
        """Analytical gradient of impact."""
        ratio = order_size / volume
        return self.coefficient * self.gamma * ratio ** (self.gamma - 1) * (1 / volume)


class LiquidityBarrier:
    """
    Models liquidity barriers where execution becomes discontinuous.
    """
    
    def __init__(self, barrier_depth: float = 0.01):
        self.barrier_depth = barrier_depth
        
    def barrier_function(self, position: float, volume: float) -> float:
        """
        Liquidity barrier function.
        Returns 1 if position exceeds barrier, 0 otherwise (smoothed).
        """
        # Smoothed step function
        position_pct = position / volume
        k = 100  # Smoothing parameter
        barrier = 1 / (1 + np.exp(-k * (position_pct - self.barrier_depth)))
        return barrier
    
    def analytical_derivative(self, position: float, volume: float) -> float:
        """
        Analytical derivative of barrier function.
        """
        position_pct = position / volume
        k = 100
        exp_term = np.exp(-k * (position_pct - self.barrier_depth))
        derivative = k * exp_term / (1 + exp_term) ** 2
        return derivative / volume


class ExecutionCostCalculator:
    """
    Combined execution cost calculator with analytical derivatives.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.impact_model = SquareRootImpactModel(
            coefficient=config.get("impact_coeff", 0.1),
            gamma=config.get("gamma", 0.5)
        )
        self.almgren_chriss = AlmgrenChrissModel(
            temp_impact=config.get("temp_impact", 0.1),
            perm_impact=config.get("perm_impact", 0.01),
            volatility=config.get("volatility", 0.2),
            risk_aversion=config.get("risk_aversion", 0.5)
        )
        self.liquidity_barrier = LiquidityBarrier(
            barrier_depth=config.get("barrier_depth", 0.01)
        )
    
    def calculate_total_cost(self, order_size: float, volume: float,
                            volatility: float, position: float) -> Dict:
        """
        Calculate total execution cost with all components.
        """
        # Price impact
        impact = self.impact_model.impact(order_size, volume)
        
        # Almgren-Chriss cost
        ac_cost = self.almgren_chriss.execution_cost(order_size, volume)
        
        # Liquidity barrier
        barrier = self.liquidity_barrier.barrier_function(position, volume)
        
        # Total cost = impact + AC cost + barrier penalty
        total_cost = impact + ac_cost + barrier * 0.01  # 1% penalty for barrier breach
        
        return {
            "impact": impact,
            "ac_cost": ac_cost,
            "barrier": barrier,
            "total_cost": total_cost
        }
    
    def analytical_gradients(self, order_size: float, volume: float,
                            volatility: float, position: float) -> Dict:
        """
        Analytical gradients of all cost components.
        """
        d_impact = self.impact_model.analytical_gradient(order_size, volume)
        d_ac = self.almgren_chriss.analytical_gradient(order_size, volume)
        d_barrier = self.liquidity_barrier.analytical_derivative(position, volume)
        
        return {
            "d_impact": d_impact,
            "d_ac": d_ac,
            "d_barrier": d_barrier,
            "d_total": d_impact + d_ac["d_total"] + d_barrier * 0.01
        }
