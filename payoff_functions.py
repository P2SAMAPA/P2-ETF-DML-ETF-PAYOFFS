"""
payoff_functions.py  —  Payoff Functions with Analytical Derivatives
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DiscontinuousPayoff:
    """
    Base class for discontinuous payoff functions with analytical derivatives.
    """
    
    def __init__(self, barrier_level: float = 0.05):
        self.barrier_level = barrier_level
    
    def payoff(self, price: torch.Tensor) -> torch.Tensor:
        """Calculate payoff."""
        raise NotImplementedError
    
    def analytical_derivative(self, price: torch.Tensor) -> torch.Tensor:
        """Analytical derivative of payoff."""
        raise NotImplementedError


class BarrierOptionPayoff(DiscontinuousPayoff):
    """
    Barrier option payoff with discontinuous derivative.
    """
    
    def __init__(self, barrier_level: float = 0.05, rebate: float = 0.5):
        super().__init__(barrier_level)
        self.rebate = rebate
        
    def payoff(self, price: torch.Tensor) -> torch.Tensor:
        """
        Payoff = max(price - barrier, 0) if price > barrier else rebate.
        """
        # Smoothed barrier for differentiability
        k = 100  # Smoothing parameter
        barrier_indicator = torch.sigmoid(k * (price - self.barrier_level))
        
        base_payoff = torch.relu(price - self.barrier_level)
        rebate_payoff = self.rebate * torch.ones_like(price)
        
        return barrier_indicator * base_payoff + (1 - barrier_indicator) * rebate_payoff
    
    def analytical_derivative(self, price: torch.Tensor) -> torch.Tensor:
        """
        Analytical derivative of barrier payoff.
        """
        k = 100
        barrier_indicator = torch.sigmoid(k * (price - self.barrier_level))
        
        # Derivative of sigmoid
        d_indicator = k * barrier_indicator * (1 - barrier_indicator)
        
        # Derivative of payoff = indicator + price * d_indicator - barrier * d_indicator - rebate * d_indicator
        d_payoff = barrier_indicator + (price - self.barrier_level - self.rebate) * d_indicator
        
        return d_payoff


class KnockoutPayoff(DiscontinuousPayoff):
    """
    Knockout option payoff with discontinuous derivative.
    """
    
    def __init__(self, barrier_level: float = 0.05, knockin_level: float = 0.10):
        super().__init__(barrier_level)
        self.knockin_level = knockin_level
        
    def payoff(self, price: torch.Tensor) -> torch.Tensor:
        """
        Payoff = 0 if price > knockin_level else max(price - barrier, 0).
        """
        k = 100
        knockout_indicator = 1 - torch.sigmoid(k * (price - self.knockin_level))
        base_payoff = torch.relu(price - self.barrier_level)
        
        return knockout_indicator * base_payoff
    
    def analytical_derivative(self, price: torch.Tensor) -> torch.Tensor:
        """
        Analytical derivative of knockout payoff.
        """
        k = 100
        sig = torch.sigmoid(k * (price - self.knockin_level))
        d_sig = k * sig * (1 - sig)
        
        knockout = 1 - sig
        d_knockout = -d_sig
        
        base_payoff = torch.relu(price - self.barrier_level)
        d_base = (price > self.barrier_level).float()
        
        return d_knockout * base_payoff + knockout * d_base


class DiscontinuityAwareLoss(nn.Module):
    """
    Loss function that accounts for payoff discontinuities.
    """
    
    def __init__(self, barrier_level: float = 0.05, 
                 discontinuity_weight: float = 0.1):
        super().__init__()
        self.barrier_level = barrier_level
        self.discontinuity_weight = discontinuity_weight
        self.barrier_payoff = BarrierOptionPayoff(barrier_level)
        
    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                price: torch.Tensor) -> Dict:
        """
        Compute loss with discontinuity-aware penalty.
        """
        # Standard MSE loss
        mse_loss = F.mse_loss(prediction, target)
        
        # Payoff discontinuity loss
        payoff_pred = self.barrier_payoff.payoff(prediction)
        payoff_target = self.barrier_payoff.payoff(target)
        payoff_loss = F.mse_loss(payoff_pred, payoff_target)
        
        # Analytical derivative loss
        d_pred = self.barrier_payoff.analytical_derivative(prediction)
        d_target = self.barrier_payoff.analytical_derivative(target)
        derivative_loss = F.mse_loss(d_pred, d_target)
        
        # Combined loss
        total_loss = (mse_loss + 
                     self.discontinuity_weight * payoff_loss +
                     0.1 * derivative_loss)
        
        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "payoff_loss": payoff_loss,
            "derivative_loss": derivative_loss
        }
