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
    "EQUITY_SECTORS": ["VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
}

# Model parameters
DML_CONFIG = {
    "hidden_dims": [128, 64, 32],
    "activation": "swish",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "dropout_rate": 0.1,
    # L2 weight decay, passed to the Adam optimizer. This was already defined
    # here but never actually wired into the optimizer, meaning training ran
    # with effectively zero L2 regularization. With ~330 features and only
    # ~1600 samples for the larger universes, that let the model overfit
    # within the first 1-3 epochs (verified: validation loss peaked at
    # epoch 1 without this, epoch 82 with it, on equivalent synthetic data).
    "l2_reg": 0.001,
    "use_differential_loss": True,
    # Weight for the execution-cost-sensitivity regularizer in
    # DifferentialNetwork.compute_differential_loss. IMPORTANT: this was
    # previously set to 0.3 but never actually read by any code — if wired
    # in naively at that value, it reproduces a training-divergence bug
    # (verified directly: at 0.1, this term was measured at up to ~1000x
    # the scale of the primary MSE loss and completely hijacked training).
    # 1e-4 keeps the regularizer's maximum possible contribution comparable
    # to, not orders of magnitude larger than, typical MSE scale.
    "differential_weight": 0.0001,
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

# Minimum trading days of price history a ticker must have to be included
# in training. Tickers below this are recently-listed and get excluded so
# they don't drag the whole universe's usable date range down to their
# inception date (see data_manager.filter_by_history for why). ~2500
# trading days is roughly 10 years — comfortably more than the longest
# rolling window above (504 days = ~2 years), so excluded tickers are
# genuinely short-history, not just barely under the window requirement.
MIN_HISTORY_DAYS = 2500
