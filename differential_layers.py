"""
differential_layers.py  —  Differential Layers for Discontinuous Payoffs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DifferentialLayer(nn.Module):
    """
    Base layer with analytical derivatives for execution costs.
    """
    
    def __init__(self):
        super().__init__()
        self._cache = {}
    
    def analytical_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Override this with analytical derivative."""
        raise NotImplementedError


class DiscontinuityAwareLinear(nn.Module):
    """
    Linear layer with discontinuity-aware activation.
    Handles tick size barriers and spread discontinuities.
    """
    
    def __init__(self, in_features: int, out_features: int, tick_size: float = 0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.tick_size = tick_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear transformation: output = x @ W^T + bias
        out = F.linear(x, self.weight, self.bias)
        
        # Apply tick size discretization (with analytical derivative)
        out = self._tick_discretize(out)
        
        return out
    
    def _tick_discretize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discretize to tick size with analytical derivative.
        Uses soft rounding for differentiability.
        """
        temperature = 0.01
        rounded = x + temperature * torch.sin(2 * np.pi * x / self.tick_size) / (2 * np.pi)
        return rounded
    
    def analytical_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analytical derivative of the tick discretization.
        d/dx [x + (tick/2π) * sin(2πx/tick)] = 1 + cos(2πx/tick)
        """
        return 1 + torch.cos(2 * np.pi * x / self.tick_size)


class PriceImpactLayer(nn.Module):
    """
    Price impact model with analytical derivatives.
    Implements square-root impact model.
    """
    
    def __init__(self, impact_coefficient: float = 0.1, gamma: float = 0.5):
        super().__init__()
        self.impact_coefficient = nn.Parameter(torch.tensor(impact_coefficient))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        
    def forward(self, order_size: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        """
        Calculate price impact using square-root model.
        impact = coeff * (order_size / volume)^gamma
        """
        ratio = order_size / (volume + 1e-8)
        impact = self.impact_coefficient * (ratio ** self.gamma)
        return impact
    
    def analytical_derivative(self, order_size: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        """
        Analytical derivative of square-root impact.
        d/d(order_size) [coeff * (order_size/volume)^gamma] 
        = coeff * gamma * (order_size/volume)^(gamma-1) * (1/volume)
        """
        ratio = order_size / (volume + 1e-8)
        return self.impact_coefficient * self.gamma * (ratio ** (self.gamma - 1)) * (1 / (volume + 1e-8))


class SpreadLayer(nn.Module):
    """
    Bid-ask spread model with regime switching.
    """
    
    def __init__(self, base_spread: float = 0.001, volatility_scaling: float = 0.5):
        super().__init__()
        self.base_spread = nn.Parameter(torch.tensor(base_spread))
        self.vol_scaling = nn.Parameter(torch.tensor(volatility_scaling))
        
    def forward(self, volatility: torch.Tensor) -> torch.Tensor:
        """
        Calculate spread with volatility scaling.
        spread = base_spread + vol_scaling * volatility
        """
        spread = self.base_spread + self.vol_scaling * volatility
        return spread.clamp(min=0.0001)
    
    def analytical_derivative(self, volatility: torch.Tensor) -> torch.Tensor:
        """
        Analytical derivative of spread with respect to volatility.
        d/d(vol) [base_spread + vol_scaling * vol] = vol_scaling
        """
        return self.vol_scaling * torch.ones_like(volatility)


class ExecutionCostLayer(nn.Module):
    """
    Total execution cost layer combining impact and spread.
    """
    
    def __init__(self, impact_coeff: float = 0.1, gamma: float = 0.5, 
                 base_spread: float = 0.001, vol_scaling: float = 0.5):
        super().__init__()
        self.impact_layer = PriceImpactLayer(impact_coeff, gamma)
        self.spread_layer = SpreadLayer(base_spread, vol_scaling)
        
    def forward(self, order_size: torch.Tensor, volume: torch.Tensor, 
                volatility: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate total execution cost and components.
        """
        impact = self.impact_layer(order_size, volume)
        spread = self.spread_layer(volatility)
        
        # Total cost = impact + half-spread
        total_cost = impact + 0.5 * spread
        
        components = {
            "impact": impact,
            "spread": spread,
            "total_cost": total_cost
        }
        
        return total_cost, components
    
    def analytical_gradient(self, order_size: torch.Tensor, volume: torch.Tensor,
                           volatility: torch.Tensor) -> Dict:
        """
        Compute analytical gradients of cost components.
        """
        d_impact = self.impact_layer.analytical_derivative(order_size, volume)
        d_spread = self.spread_layer.analytical_derivative(volatility)
        
        return {
            "d_impact": d_impact,
            "d_spread": d_spread,
            "d_total": d_impact + 0.5 * d_spread
        }


class DifferentialNetwork(nn.Module):
    """
    Neural network with differential execution cost layers.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1, config: Dict = None):
        super().__init__()
        config = config or {}
        
        hidden_dims = config.get("hidden_dims", [128, 64, 32])
        dropout_rate = config.get("dropout_rate", 0.1)
        
        self.tick_size = 0.01
        
        # Build layers - ensure first layer matches input_dim
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(DiscontinuityAwareLinear(prev_dim, hidden_dim, self.tick_size))
            layers.append(nn.SiLU())  # Swish activation
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer - maps from last hidden dim to output_dim
        self.output_layer = DiscontinuityAwareLinear(prev_dim, output_dim, self.tick_size)
        
        # Execution cost layer
        self.execution_layer = ExecutionCostLayer(
            impact_coeff=config.get("impact_coeff", 0.1),
            gamma=config.get("gamma", 0.5),
            base_spread=config.get("base_spread", 0.001)
        )
        
    def forward(self, x: torch.Tensor, volume: Optional[torch.Tensor] = None,
                volatility: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with execution cost adjustments.
        """
        # Base prediction
        h = self.hidden_layers(x)
        prediction = self.output_layer(h)
        
        # Apply execution costs if volume and volatility provided
        if volume is not None and volatility is not None:
            # Calculate execution cost
            order_size = torch.abs(prediction)  # Simplified
            cost, components = self.execution_layer(order_size, volume, volatility)
            
            # Adjust prediction for execution costs
            adjusted_prediction = prediction - cost
            
            return adjusted_prediction, {
                "base_prediction": prediction,
                "execution_cost": cost,
                "components": components,
                "adjusted_prediction": adjusted_prediction
            }
        
        return prediction, {"base_prediction": prediction}
    
    def compute_differential_loss(self, prediction: torch.Tensor, 
                                  target: torch.Tensor,
                                  volume: torch.Tensor,
                                  volatility: torch.Tensor) -> Dict:
        """
        Compute loss with differential execution cost gradients.
        """
        # Standard MSE loss
        mse_loss = F.mse_loss(prediction, target)
        
        # Analytical execution cost gradient
        order_size = torch.abs(prediction)
        gradients = self.execution_layer.analytical_gradient(
            order_size, volume, volatility
        )
        
        # Differential loss component
        # Penalize high execution cost sensitivity
        diff_loss = torch.mean(gradients["d_total"] ** 2)
        
        # Combined loss
        total_loss = mse_loss + 0.1 * diff_loss
        
        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "diff_loss": diff_loss,
            "gradients": gradients
        }
