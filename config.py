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

# Windows for training. Tied directly to PREDICTION_HORIZON_DAYS below: a
# single window of the SAME length as the forecast horizon ("does the past
# 21 days' momentum/vol predict the next 21 days' return") is a principled
# choice, not an arbitrary one, and keeps feature count in check.
#
# FIX: previously [63, 126, 252, 504] (4 windows) gave larger universes far
# more features than they had genuinely independent training examples for.
# With a 21-day forecast horizon, adjacent training ROWS overlap ~95% (they
# share 20 of 21 target days), so the count of effectively-independent
# samples is ~total_samples/horizon, not total_samples. Verified on the
# actual EQUITY_SECTORS run: ~171 effective independent samples vs. 307
# features with 4 windows (ratio 0.56 -- fewer independent samples than
# features, a mathematically underdetermined regime that explains the
# instant-overfit collapse). With a single 21-day window: 109 features,
# ratio 1.57 -- COMBINED goes from 0.46 to 1.32. FI_COMMODITIES (few
# tickers) was already fine either way (2.51 -> 5.90, more margin still).
WINDOWS = [21]

# Target ratio of (effective independent samples) / (feature count) used to
# scale L2/dropout regularization per-universe in trainer.py -- universes
# that fall short of this target (typically the larger ones) get extra
# regularization pressure proportional to how far short they are, up to
# MAX_REG_SCALE. Universes already at or above this ratio (e.g.
# FI_COMMODITIES) are left at the base l2_reg/dropout_rate.
TARGET_EFFECTIVE_RATIO = 2.0
MAX_REG_SCALE = 5.0

# Minimum trading days of price history a ticker must have to be included
# in training. Tickers below this are recently-listed and get excluded so
# they don't drag the whole universe's usable date range down to their
# inception date (see data_manager.filter_by_history for why). Raised from
# 2500 (~10y) to 3500 (~14y): at 2500, EQUITY_SECTORS/COMBINED still only
# reached ~2331 usable samples (~9.25y) because a ~2015-vintage ETF cleared
# the bar just barely and became the new binding constraint. 3500 trading
# days requires a ~2012 or earlier listing date, pushing the common start
# back toward the ~2004-2006 vintage sector ETFs that make up most of the
# universe, much closer to FI_COMMODITIES's ~15-year window.
MIN_HISTORY_DAYS = 3500

# Forecast horizon in trading days. The target becomes the cumulative
# return over the NEXT this-many trading days (e.g. 21 ~= 1 trading month),
# instead of just tomorrow's return. Longer horizons carry more signal
# relative to daily noise, which is what lets the model's z-score-based
# confidence (see trainer.py's generate_picks) actually reach Medium/High
# rather than being structurally capped near 0 on 1-day-ahead noise.
PREDICTION_HORIZON_DAYS = 21

# How many top picks to surface per universe.
TOP_N_PICKS = 1
