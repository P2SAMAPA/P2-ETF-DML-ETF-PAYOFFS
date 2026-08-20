"""
config.py  —  Configuration for Differential Machine Learning
"""

import os

# HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-dml-etf-payoffs-results"

# Universes
UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
}

# Model parameters
DML_CONFIG = {
    "hidden_dims": [128, 64, 32],
    "activation": "swish",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "dropout_rate": 0.1,
    "l2_reg": 0.001,
    "use_differential_loss": True,
    "differential_weight": 0.3,
}

# Execution parameters
EXECUTION_CONFIG = {
    "impact_model": "square_root",
    "spread_model": "constant",
    "max_position_pct": 0.02,
    "min_tick_size": 0.01,
    "execution_horizon": 5,
    "risk_aversion": 0.5,
}

# Windows for training
WINDOWS = [63, 126, 252, 504]
