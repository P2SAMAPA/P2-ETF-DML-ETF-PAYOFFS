"""
trainer.py  —  Differential Machine Learning Trainer
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, prepare_features
from differential_layers import DifferentialNetwork
from execution_models import ExecutionCostCalculator
from payoff_functions import DiscontinuityAwareLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_trainer() -> Dict:
    """Main training orchestrator."""
    
    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}
    
    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {
        "run_date": run_date,
        "universes": {},
        "execution_metrics": {}
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize execution cost calculator
    exec_calc = ExecutionCostCalculator(config.EXECUTION_CONFIG)
    
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Training on {universe_name}...")
        
        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            continue
        
        # Prepare features
        X, y, volumes, volatilities = prepare_features(
            prices_df[available], macro_df
        )
        
        if len(X) < 100:
            logger.warning(f"Not enough data for {universe_name}")
            continue
        
        # Convert to tensors
        X_t = torch.FloatTensor(X).to(device)
        y_t = torch.FloatTensor(y).reshape(-1, 1).to(device)
        V_t = torch.FloatTensor(volumes).reshape(-1, 1).to(device)
        Vol_t = torch.FloatTensor(volatilities).reshape(-1, 1).to(device)
        
        # Create dataset
        dataset = TensorDataset(X_t, y_t, V_t, Vol_t)
        dataloader = DataLoader(dataset, batch_size=config.DML_CONFIG["batch_size"], shuffle=True)
        
        # Initialize model
        model = DifferentialNetwork(
            input_dim=X.shape[1],
            output_dim=1,
            config=config.DML_CONFIG
        ).to(device)
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=config.DML_CONFIG["learning_rate"])
        loss_fn = DiscontinuityAwareLoss(barrier_level=0.05)
        
        # Training loop
        logger.info(f"  Training {len(dataloader)} batches...")
        model.train()
        
        epoch_losses = []
        for epoch in range(config.DML_CONFIG["epochs"]):
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_diff = 0.0
            
            for batch_idx, (X_batch, y_batch, V_batch, Vol_batch) in enumerate(dataloader):
                optimizer.zero_grad()
                
                # Forward pass with execution costs
                prediction, info = model(X_batch, V_batch, Vol_batch)
                
                # Compute differential loss
                loss_dict = model.compute_differential_loss(
                    prediction, y_batch, V_batch, Vol_batch
                )
                
                loss = loss_dict["total_loss"]
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_mse += loss_dict["mse_loss"].item()
                epoch_diff += loss_dict["diff_loss"].item()
            
            avg_loss = epoch_loss / len(dataloader)
            avg_mse = epoch_mse / len(dataloader)
            avg_diff = epoch_diff / len(dataloader)
            epoch_losses.append(avg_loss)
            
            if epoch % 20 == 0:
                logger.info(f"    Epoch {epoch}: Loss={avg_loss:.6f}, MSE={avg_mse:.6f}, Diff={avg_diff:.6f}")
        
        # Store results
        results["universes"][universe_name] = {
            "loss": avg_loss,
            "mse": avg_mse,
            "diff_loss": avg_diff,
            "tickers": available
        }
    
    # Save results
    output_path = f"dml_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Saved: {output_path}")
    
    return results


if __name__ == "__main__":
    run_trainer()
