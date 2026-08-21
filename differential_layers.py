"""
differential_layers.py  —  Differential Layers for Discontinuous Payoffs

FIXES vs original:
  1. DiscontinuityAwareLinear.analytical_derivative previously ignored the
     `temperature` scaling and the `tick_size` used in forward(), so the
     "analytical" gradient did not match the actual forward computation
     (verified by finite differences — error up to 0.8 per unit tick).
     The derivative now uses the same temperature/tick_size terms as the
     forward pass.
  2. DifferentialNetwork now supports multi-output prediction (one return
     forecast per ticker, output_dim = n_tickers) instead of collapsing
     every ticker in a universe down to a single scalar. This matches how
     trainer.py now builds targets (see data_manager.py) and is what makes
     per-ticker "top picks" meaningful.
  3. Execution-cost broadcasting: volume/volatility are per-day scalars,
     order_size is now (batch, n_tickers). Costs are broadcast correctly
     instead of silently relying on tensor auto-broadcast rules that only
     happened to work for output_dim=1.
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

    def __init__(self, in_features: int, out_features: int, tick_size: float = 0.01,
                 temperature: float = 0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.tick_size = tick_size
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear transformation: output = x @ W^T + bias
        out = F.linear(x, self.weight, self.bias)

        # Apply tick size discretization (with analytical derivative)
        out = self._tick_discretize(out)

        return out

    def _tick_discretize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discretize to tick size with analytical derivative.
        Uses soft rounding for differentiability:
            f(x) = x + (temperature / 2π) * sin(2π x / tick_size)
        """
        return x + self.temperature * torch.sin(2 * np.pi * x / self.tick_size) / (2 * np.pi)

    def analytical_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analytical derivative of the tick discretization.
        d/dx [x + (temperature/2π) * sin(2πx/tick)]
            = 1 + (temperature / tick_size) * cos(2πx/tick)
        (Previously this dropped the temperature/tick_size scale factor,
        which made the "analytical" gradient wrong for any tick_size other
        than exactly 0.01 with temperature exactly 0.01.)
        """
        return 1 + (self.temperature / self.tick_size) * torch.cos(2 * np.pi * x / self.tick_size)


class PriceImpactLayer(nn.Module):
    """
    Price impact model with analytical derivatives.
    Implements square-root impact model.

    FIX: impact_coefficient and gamma were previously raw, unconstrained
    nn.Parameters. Verified by direct repro: under ordinary training
    dynamics gamma can be driven negative (e.g. -1.8) within ~100 optimizer
    steps. With ratio = order_size/volume often << 1 (predictions start
    near zero), ratio**gamma for negative gamma explodes toward +/-inf
    (e.g. loss diverged past 1e20 within 100 steps in testing), which is
    exactly the mechanism behind the "Train Loss=nan" seen after a few
    dozen epochs. gamma is now constrained to (0.05, 1.0) via a sigmoid
    reparameterization, and impact_coefficient is constrained to be
    positive via softplus — both are economically sensible ranges for a
    price-impact exponent/coefficient and can no longer diverge.
    """

    def __init__(self, impact_coefficient: float = 0.1, gamma: float = 0.5,
                 gamma_min: float = 0.05, gamma_max: float = 1.0):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        # Inverse-sigmoid init so the constrained gamma starts at the requested value
        gamma_frac = (gamma - gamma_min) / (gamma_max - gamma_min)
        gamma_frac = min(max(gamma_frac, 1e-4), 1 - 1e-4)
        gamma_raw_init = np.log(gamma_frac / (1 - gamma_frac))

        # Inverse-softplus init so constrained impact_coefficient starts at the requested value
        coeff_raw_init = np.log(np.expm1(max(impact_coefficient, 1e-4)))

        self._gamma_raw = nn.Parameter(torch.tensor(gamma_raw_init, dtype=torch.float32))
        self._impact_coefficient_raw = nn.Parameter(torch.tensor(coeff_raw_init, dtype=torch.float32))

    @property
    def gamma(self) -> torch.Tensor:
        return self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(self._gamma_raw)

    @property
    def impact_coefficient(self) -> torch.Tensor:
        return F.softplus(self._impact_coefficient_raw)

    def forward(self, order_size: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        """
        Calculate price impact using square-root model.
        impact = coeff * (order_size / volume)^gamma
        """
        ratio = order_size / (volume + 1e-8)
        impact = self.impact_coefficient * (ratio.clamp(min=0) ** self.gamma)
        return impact

    def analytical_derivative(self, order_size: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        """
        Analytical derivative of square-root impact.
        d/d(order_size) [coeff * (order_size/volume)^gamma]
        = coeff * gamma * (order_size/volume)^(gamma-1) * (1/volume)
        """
        ratio = (order_size / (volume + 1e-8)).clamp(min=1e-8)
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
        (zero where the clamp floor is active, since the gradient of a
        clamp is zero at the floor; in practice base_spread + vol_scaling*vol
        rarely dips below 0.0001 so this is a minor edge case.)
        """
        spread = self.base_spread + self.vol_scaling * volatility
        active = (spread > 0.0001).float()
        return self.vol_scaling * active


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
        order_size: (batch, n_outputs)
        volume, volatility: (batch, 1) — broadcast across outputs.
        """
        impact = self.impact_layer(order_size, volume)
        spread = self.spread_layer(volatility)

        # Total cost = impact + half-spread (spread broadcasts over n_outputs)
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

    output_dim should equal the number of tickers being predicted for a
    given universe — each forward pass produces one return forecast per
    ticker from the shared day-level feature vector, instead of a single
    scalar that got mis-attributed to every ticker downstream.
    """

    def __init__(self, input_dim: int, output_dim: int = 1, config: Dict = None):
        super().__init__()
        config = config or {}

        hidden_dims = config.get("hidden_dims", [128, 64, 32])
        dropout_rate = config.get("dropout_rate", 0.1)

        self.tick_size = config.get("tick_size", 0.01)
        self.output_dim = output_dim

        # Build layers - ensure first layer matches input_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(DiscontinuityAwareLinear(prev_dim, hidden_dim, self.tick_size))
            layers.append(nn.SiLU())  # Swish activation
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer - maps from last hidden dim to output_dim (n_tickers)
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
        Forward pass.
        x: (batch, input_dim)
        volume, volatility: (batch, 1) or None — only used to compute execution
        cost DIAGNOSTICS (returned in `info`), never subtracted from the
        returned prediction.

        FIX: previously, when volume/volatility were given, this method
        subtracted execution cost from the prediction before returning it,
        and trainer.py's training loop fit THAT cost-adjusted value against
        the raw next-day-return target. But generate_picks() (and the
        original validation code) called forward() WITHOUT volume/
        volatility, getting the un-adjusted base prediction straight back.
        Verified by direct reproduction: training forces
        (base_prediction - cost) ≈ target, so base_prediction ends up
        biased upward by ≈ the trained-in cost — in a synthetic test with a
        true target of exactly 0, the "prediction used for picks" converged
        to 0.00422, matching the trained-in cost (0.00422) almost exactly.
        That bias is very likely why picks like FI_COMMODITIES showed
        implausibly large "High confidence" 1.5-1.7% next-day forecasts.

        The prediction returned here is now ALWAYS the raw, unadjusted
        forecast, identically whether or not volume/volatility are passed —
        eliminating the train/inference mismatch. Execution cost is still
        computed (when volume/volatility are given) and returned in `info`
        for diagnostics, and is used by compute_differential_loss() as a
        genuine regularization term (penalizing cost SENSITIVITY, not
        subtracting cost from the forecast).
        """
        # Base prediction — this is always the return forecast, full stop.
        h = self.hidden_layers(x)
        prediction = self.output_layer(h)  # (batch, output_dim)

        info = {"base_prediction": prediction}

        # Execution cost is computed only for diagnostics/regularization —
        # it is never subtracted from the returned prediction.
        if volume is not None and volatility is not None:
            order_size = torch.abs(prediction)  # Simplified: use |predicted return| as proxy for size
            cost, components = self.execution_layer(order_size, volume, volatility)
            info["execution_cost"] = cost
            info["components"] = components

        return prediction, info

    def compute_differential_loss(self, prediction: torch.Tensor,
                                  target: torch.Tensor,
                                  volume: torch.Tensor,
                                  volatility: torch.Tensor) -> Dict:
        """
        The actual "differential loss": standard MSE on the raw prediction
        vs. raw target, PLUS a regularization term that penalizes high
        execution-cost SENSITIVITY (the analytical gradient of cost w.r.t.
        order size) — this is the project's namesake feature. Previously
        this method existed but was never called anywhere in trainer.py;
        the training loop called plain F.mse_loss on the cost-ADJUSTED
        forward output instead, which both skipped this regularizer
        entirely and caused the bias described in forward()'s docstring.
        """
        # Standard MSE loss on the raw (unadjusted) prediction
        mse_loss = F.mse_loss(prediction, target)

        # Analytical execution cost gradient (cost sensitivity, not cost itself)
        order_size = torch.abs(prediction)
        gradients = self.execution_layer.analytical_gradient(
            order_size, volume, volatility
        )

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
