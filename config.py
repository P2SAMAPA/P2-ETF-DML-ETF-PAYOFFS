"""
config.py  —  Configuration for Differential Machine Learning
"""

import os

# HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN")
RESULTS_REPO = "P2SAMAPA/p2-dml-etf-payoffs-results"
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"

# Universes
UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB","IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB"],
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
    "impact_model": "square_root",  # "linear", "square_root", "almgren_chriss"
    "spread_model": "constant",      # "constant", "volatility_based", "regime_switching"
    "max_position_pct": 0.02,        # Max position as % of ADV
    "min_tick_size": 0.01,           # Minimum price increment
    "execution_horizon": 5,          # Days to execute
    "risk_aversion": 0.5,            # Risk aversion parameter
}

# Windows for training
WINDOWS = [63, 126, 252, 504]
