"""
trainer.py  —  Differential Machine Learning Trainer
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data, prepare_features
from differential_layers import DifferentialNetwork
from execution_models import ExecutionCostCalculator
from payoff_functions import DiscontinuityAwareLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_picks(model, X: np.ndarray, tickers: List[str], 
                   top_n: int = 3) -> List[Dict]:
    """Generate top N ETF picks from model predictions."""
    device = next(model.parameters()).device
    
    # Convert to tensor
    X_t = torch.FloatTensor(X).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions, _ = model(X_t, None, None)
    
    # Convert to numpy
    preds = predictions.cpu().numpy().flatten()
    
    # Create list of (ticker, prediction) pairs
    ticker_preds = list(zip(tickers, preds))
    
    # Sort by prediction (highest first)
    sorted_picks = sorted(ticker_preds, key=lambda x: x[1], reverse=True)
    
    # Take top N
    top_picks = sorted_picks[:top_n]
    
    # Format results
    results = []
    for ticker, pred in top_picks:
        results.append({
            "ticker": ticker,
            "expected_return": round(float(pred) * 100, 2),  # Convert to percentage
            "confidence": "High" if pred > 0.01 else "Medium" if pred > 0.005 else "Low",
            "rationale": f"DML prediction: {float(pred):.4f}"
        })
    
    return results


def run_trainer() -> Dict:
    """Main training orchestrator."""
    
    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}
    
    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {
        "run_date": run_date,
        "universes": {},
        "execution_metrics": {},
        "top_picks": {}
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize execution cost calculator
    exec_calc = ExecutionCostCalculator(config.EXECUTION_CONFIG)
    
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Training on {universe_name}...")
        
        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            logger.warning(f"No tickers available for {universe_name}")
            continue
        
        # Prepare features using prices
        try:
            X, y, volumes, volatilities = prepare_features(
                prices_df[available], macro_df
            )
        except Exception as e:
            logger.error(f"Failed to prepare features for {universe_name}: {e}")
            continue
        
        # Check shapes
        logger.info(f"Feature shape: {X.shape}, Target shape: {y.shape}")
        
        if len(X) != len(y):
            logger.error(f"Shape mismatch: X={X.shape}, y={y.shape}")
            continue
        
        if len(X) < 100:
            logger.warning(f"Not enough data for {universe_name}: {len(X)} samples")
            continue
        
        # Split into train and validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        V_train, V_val = volumes[:split_idx], volumes[split_idx:]
        Vol_train, Vol_val = volatilities[:split_idx], volatilities[split_idx:]
        
        # Convert to tensors
        X_t = torch.FloatTensor(X_train).to(device)
        y_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
        V_t = torch.FloatTensor(V_train).reshape(-1, 1).to(device)
        Vol_t = torch.FloatTensor(Vol_train).reshape(-1, 1).to(device)
        
        # Validation tensors
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
        
        # Create dataset
        dataset = TensorDataset(X_t, y_t, V_t, Vol_t)
        dataloader = DataLoader(dataset, batch_size=config.DML_CONFIG["batch_size"], shuffle=True)
        
        # Initialize model
        model = DifferentialNetwork(
            input_dim=X.shape[1],
            output_dim=1,
            config=config.DML_CONFIG
        ).to(device)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.DML_CONFIG["learning_rate"])
        
        # Training loop
        logger.info(f"  Training {len(dataloader)} batches for {config.DML_CONFIG['epochs']} epochs...")
        model.train()
        
        best_val_loss = float('inf')
        epoch_losses = []
        
        for epoch in range(config.DML_CONFIG["epochs"]):
            epoch_loss = 0.0
            
            for X_batch, y_batch, V_batch, Vol_batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                prediction, info = model(X_batch, V_batch, Vol_batch)
                
                # MSE loss only for training stability
                loss = F.mse_loss(prediction, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred, _ = model(X_val_t, None, None)
                val_loss = F.mse_loss(val_pred, y_val_t).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            if epoch % 20 == 0:
                logger.info(f"    Epoch {epoch}: Train Loss={avg_loss:.6f}, Val Loss={val_loss:.6f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Generate picks using validation set (latest data)
        picks = generate_picks(model, X_val[-50:], available, top_n=3)
        
        # Store results
        results["universes"][universe_name] = {
            "loss": avg_loss,
            "best_val_loss": best_val_loss,
            "tickers": available,
            "samples": len(X)
        }
        
        results["top_picks"][universe_name] = picks
        
        logger.info(f"  ✅ Top picks for {universe_name}:")
        for pick in picks:
            logger.info(f"     {pick['ticker']}: {pick['expected_return']}% ({pick['confidence']})")
    
    # Save results
    output_path = f"dml_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Saved: {output_path}")
    
    # Upload to HuggingFace
    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")
    
    return results


if __name__ == "__main__":
    run_trainer()
