"""
data_manager.py  —  Data loading and feature preparation

FIX vs original (prepare_features):
  1. The old version collapsed every ticker's next-day return into a single
     cross-sectional mean (`y_mean`), so the model only ever learned to
     predict "tomorrow's average return across the whole universe" — one
     number per day. trainer.py then zipped that one number-per-day against
     the list of *tickers* to produce "top picks", which is a mismatch: it
     was never predicting anything ticker-specific. Fixed here by returning
     a full (n_samples, n_tickers) target matrix Y — one next-day return
     per ticker — so the model can actually learn per-ticker structure and
     "top picks" are meaningful.
  2. Feature normalization was previously done in this function using the
     full sample's mean/std (including what becomes the validation split
     later in trainer.py) — a train/validation leakage bug. Normalization
     is now the caller's responsibility (trainer.py fits it on the train
     split only and applies the same transform to validation).
"""

import os
import sys
import logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = logging.getLogger(__name__)

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
MACRO_COLS_ALL = MACRO_COLS_CORE + MACRO_COLS_EXTENDED


def _all_tickers() -> List[str]:
    seen, result = set(), []
    for tickers in config.UNIVERSES.values():
        for t in tickers:
            if t not in seen:
                seen.add(t)
                result.append(t)
    return result


def load_master_data(hf_token: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    token = hf_token or config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError("HF_TOKEN not set")

    api = HfApi(token=token)
    logger.info(f"Downloading master parquet from {config.DATA_REPO} …")

    parquet_path = None
    for fname in ["master_data.parquet", "data/master.parquet", "master.parquet"]:
        try:
            parquet_path = api.hf_hub_download(repo_id=config.DATA_REPO, filename=fname, repo_type="dataset", token=token)
            logger.info(f"  → found at '{fname}'")
            break
        except Exception:
            continue

    if parquet_path is None:
        raise RuntimeError(f"Could not locate master parquet in {config.DATA_REPO}.")

    df = pd.read_parquet(parquet_path)
    logger.info(f"Raw parquet: {df.shape[0]} rows × {df.shape[1]} cols")

    if not isinstance(df.index, pd.DatetimeIndex):
        for date_col in ["Date", "date", "DATE", "timestamp"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                break
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df.index.name = "Date"

    all_tickers = _all_tickers()
    avail_tickers = [t for t in all_tickers if t in df.columns]
    if not avail_tickers:
        raise RuntimeError("No ETF ticker columns found.")

    prices = df[avail_tickers].copy()
    prices = prices.ffill()
    prices = prices.dropna(how="all")

    avail_core = [c for c in MACRO_COLS_CORE if c in df.columns]
    avail_ext = [c for c in MACRO_COLS_EXTENDED if c in df.columns]
    avail_all = avail_core + avail_ext

    if not avail_all:
        raise RuntimeError("No macro columns found.")

    macro = df[avail_all].copy()

    if avail_core:
        before = len(macro)
        macro = macro.dropna(subset=avail_core)
        dropped = before - len(macro)
        if dropped:
            logger.info(f"Dropped {dropped} rows with NaN in core macro cols.")

    if avail_ext:
        macro[avail_ext] = macro[avail_ext].ffill().fillna(0.0)

    common = prices.index.intersection(macro.index)
    if len(common) == 0:
        raise RuntimeError("No overlapping dates.")

    prices = prices.loc[common]
    macro = macro.loc[common]

    logger.info(f"Dataset ready: {len(prices)} rows | {len(avail_tickers)} ETFs | {len(avail_all)} macro cols")
    return prices, macro


def validate_data(prices: pd.DataFrame, macro: pd.DataFrame) -> None:
    for ticker in list(prices.columns)[:3]:
        col = prices[ticker].dropna()
        if len(col) > 10 and abs(col.median()) < 0.05:
            logger.warning(f"'{ticker}' median ≈ {col.median():.4f} — looks like returns, not prices!")

    if len(prices) < 252:
        logger.warning(f"Only {len(prices)} rows — less than 1 year of data.")


def prepare_features(prices_df: pd.DataFrame, macro_df: pd.DataFrame) -> tuple:
    """
    Prepare features for DML training.

    Returns:
        X:          (n_samples, n_tickers) — today's log returns for every
                    ticker in the universe (unnormalized; caller normalizes
                    using train-split statistics to avoid leakage).
        Y:          (n_samples, n_tickers) — next-day log return for EACH
                    ticker (not the cross-sectional mean), so a model with
                    output_dim = n_tickers predicts one number per ticker.
        volumes:    (n_samples,) proxy volume for the day.
        volatility: (n_samples,) proxy realized volatility for the day.
    """

    # Calculate log returns from prices
    returns = np.log(prices_df / prices_df.shift(1)).dropna()

    if len(returns) < 60:
        raise ValueError(f"Not enough return data: {len(returns)} rows")

    tickers = list(returns.columns)

    # Features: today's returns across all tickers in the universe
    X_full = returns.values  # (n_days, n_tickers)

    # Targets: next-day return for EACH ticker (shift -1), same shape as X
    Y_full = returns.shift(-1).values  # (n_days, n_tickers)

    # Drop the last row (target is NaN there because of the shift)
    X = X_full[:-1]
    Y = Y_full[:-1]

    # Drop any row where ANY ticker's target is NaN, so every sample has a
    # fully-populated per-ticker target vector.
    valid = ~np.isnan(Y).any(axis=1)
    X = X[valid]
    Y = Y[valid]

    if len(X) < 60:
        raise ValueError(f"Not enough valid rows after alignment: {len(X)}")

    # Volume/volatility proxies (per day, broadcast across tickers downstream)
    volumes = np.abs(X).mean(axis=1) * 1_000_000
    volatility = np.std(X, axis=1)

    return X, Y, volumes, volatility, tickers
