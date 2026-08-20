"""
data_manager.py  —  Data loading and feature preparation
"""

import os
import logging
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def load_master_data(token: str = None) -> tuple:
    """Load master data from HuggingFace."""
    try:
        local_path = hf_hub_download(
            repo_id="P2SAMAPA/fi-etf-macro-signal-master-data",
            filename="master_data.parquet",
            token=token,
            repo_type="dataset"
        )
        
        df = pd.read_parquet(local_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Separate ETFs and macro
        macro_cols = [c for c in df.columns if c.endswith('_macro')]
        price_cols = [c for c in df.columns if c not in macro_cols]
        
        prices_df = df[price_cols].copy()
        macro_df = df[macro_cols].copy()
        
        logger.info(f"✅ Loaded {len(prices_df)} days, {len(price_cols)} ETFs")
        return prices_df, macro_df
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def prepare_features(prices_df: pd.DataFrame, macro_df: pd.DataFrame) -> tuple:
    """Prepare features for DML training."""
    
    returns = np.log(prices_df / prices_df.shift(1)).dropna()
    
    # Create feature matrix
    features = []
    
    # Price features
    for col in returns.columns:
        features.append(returns[col].values)
    
    # Technical features
    momentum_20 = returns.rolling(20).mean().values
    momentum_60 = returns.rolling(60).mean().values
    volatility = returns.rolling(30).std().values
    
    # Macro features
    macro_norm = (macro_df - macro_df.mean()) / macro_df.std()
    macro_values = macro_norm.values
    
    # Combine features
    X = np.column_stack([
        returns.values,
        momentum_20,
        momentum_60,
        volatility,
        macro_values[:len(returns)]
    ])
    
    # Target: next day return
    y = returns.shift(-1).values.flatten()
    
    # Volumes (simplified - use volatility as proxy)
    volumes = np.abs(returns).values * 1000000  # Simplified volume proxy
    
    # Remove NaN
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    volumes = volumes[valid]
    volatility = volatility[valid]
    
    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    return X, y, volumes, volatility
