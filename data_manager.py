"""
data_manager.py  —  Data loading and feature preparation

FIX vs original (prepare_features):
  1. [Previous fix, kept] The old version collapsed every ticker's next-day
     return into a single cross-sectional mean, so the model only ever
     learned "tomorrow's average return across the universe" — one number
     per day — while trainer.py zipped that single number against a list
     of tickers as if it were per-ticker forecasts. Fixed by returning a
     full (n_samples, n_tickers) target matrix Y — one next-day return per
     ticker.
  2. [New] Macro data (VIX, T10Y2Y, DXY, IG_SPREAD, HY_SPREAD) was loaded
     by load_master_data() but never actually joined into the feature
     matrix — the model only ever saw same-day returns of the tickers in
     its own universe, nothing about the macro backdrop. Both macro LEVELS
     and 1-day CHANGES are now included as features (levels capture regime,
     changes capture shocks).
  3. [New] Rolling per-ticker momentum and volatility features are added
     using the trading-day windows already defined in config.WINDOWS
     (63/126/252/504 = ~1Q/6M/1Y/2Y). These were defined in config.py but
     never used anywhere in the codebase.
  4. [New] The join logic previously required EVERY ticker in a universe
     to have data on a given day (pandas .dropna() default), so a single
     late-inception ETF (e.g. XLC launched 2018, XLRE 2015, QUAL 2013)
     silently forced the ENTIRE universe's training window to start from
     that ticker's listing date — discarding up to a decade of otherwise
     usable 2008+ history for every other ticker in the universe. This is
     unavoidable in a joint multi-ticker prediction (the model needs a
     complete row to train on), but it is now surfaced explicitly via a
     log message reporting the actual usable date range and how many
     candidate rows were dropped and why, so it's visible instead of silent.
  5. [Previous fix, kept] Feature normalization is the CALLER's
     responsibility (trainer.py fits it on the train split only), so this
     function returns raw, unnormalized features.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple
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


def filter_by_history(prices_df: pd.DataFrame, tickers: List[str],
                       min_days: int) -> Tuple[List[str], Dict[str, int]]:
    """
    Split `tickers` into those with at least `min_days` of actual price
    history and those without (typically recently-listed ETFs).

    This exists because prepare_features()'s joint-availability requirement
    (a row is only usable if EVERY ticker has data that day) means a single
    late-inception ticker in a universe silently drags the entire universe's
    usable training window down to that ticker's listing date — e.g. XLC
    (listed 2018) or XLRE (listed 2015) forcing a universe with mostly
    2004+ tickers down to a ~6.5-year window. Filtering out the short-
    history tickers BEFORE calling prepare_features lets the remaining
    long-history tickers use their full available history.

    Returns:
        kept:    tickers with >= min_days of history, in original order.
        dropped: {ticker: n_days_available} for tickers excluded, so the
                 caller can log exactly what was excluded and why.
    """
    kept, dropped = [], {}
    for t in tickers:
        if t not in prices_df.columns:
            continue
        n_valid = int(prices_df[t].notna().sum())
        if n_valid >= min_days:
            kept.append(t)
        else:
            dropped[t] = n_valid
    return kept, dropped


def prepare_features(prices_df: pd.DataFrame, macro_df: pd.DataFrame,
                      windows: Optional[List[int]] = None,
                      horizon: int = 1) -> tuple:
    """
    Prepare features for DML training.

    Feature matrix X (per day t) includes:
      - Today's log return for every ticker in the universe
      - Macro LEVELS at day t (VIX, T10Y2Y, DXY, spreads if available)
      - Macro 1-day CHANGES at day t (captures shocks, not just regime)
      - Rolling per-ticker momentum (mean return) for each window in
        `windows` (default: config.WINDOWS)
      - Rolling per-ticker volatility (std of returns) for each window

    All rolling/macro features use only data up to and including day t
    (pandas .rolling()/.diff() are inherently causal), so there is no
    look-ahead leakage.

    Target Y (per day t): cumulative log return over the NEXT `horizon`
    trading days (t+1 through t+horizon) for EACH ticker — i.e. log(P_{t+horizon}/P_t).
    horizon=1 (default) reproduces the original next-day-return target.

    IMPORTANT — "latest row" for live prediction vs. training data are NOT
    the same thing once horizon > 1. A row's TARGET can only be known once
    `horizon` future days have actually happened, so the most recent ~horizon
    rows are necessarily excluded from the training set (their target is
    still in the future). But FEATURES only depend on past data, so the
    single most recent trading day's feature vector is usually available
    even though its target isn't — that's exactly the row you want to feed
    the model for a live forecast. Using the training set's last row for
    that (the old behavior) would silently forecast from data that's
    `horizon - 1` trading days stale.

    Returns:
        X:            (n_samples, n_features) unnormalized. Caller (trainer.py)
                      normalizes using train-split-only statistics.
        Y:            (n_samples, n_tickers), cumulative `horizon`-day forward return.
        volumes:      (n_samples,) proxy volume for the day, from raw returns only.
        volatility:   (n_samples,) proxy realized volatility, from raw returns only.
        tickers:      list of ticker names, in the same column order as Y.
        latest_row:   (n_features,) raw (unnormalized) feature vector for the
                      most recent trading day with complete features — for
                      live/current forecasting, NOT part of the training set.
        latest_date:  the date corresponding to latest_row (for logging).
    """
    windows = windows or config.WINDOWS
    tickers = list(prices_df.columns)

    # Daily log returns (today's return per ticker) — the base feature block
    returns = np.log(prices_df / prices_df.shift(1))

    if returns.dropna(how="all").shape[0] < 60:
        raise ValueError(f"Not enough return data: {returns.dropna(how='all').shape[0]} rows")

    feature_blocks = [returns.add_suffix("_ret")]

    # Rolling per-ticker momentum & volatility across the configured windows
    for w in windows:
        mom = returns.rolling(window=w, min_periods=w).mean().add_suffix(f"_mom{w}")
        vol = returns.rolling(window=w, min_periods=w).std().add_suffix(f"_vol{w}")
        feature_blocks.append(mom)
        feature_blocks.append(vol)

    # Macro features aligned to the same dates: levels + 1-day changes
    macro_aligned = macro_df.reindex(returns.index)
    feature_blocks.append(macro_aligned.add_suffix("_lvl"))
    feature_blocks.append(macro_aligned.diff().add_suffix("_chg"))

    feat_df = pd.concat(feature_blocks, axis=1)

    # Target: cumulative return over the NEXT `horizon` trading days.
    # shifted.iloc[t] = returns.iloc[t+1] (tomorrow's return, at row t)
    # Then a REVERSED trailing-window sum turns that into a forward-looking
    # sum: target.iloc[t] = sum(shifted.iloc[t : t+horizon])
    #                     = sum(returns.iloc[t+1 : t+horizon+1])
    #                     = log(P_{t+horizon} / P_t)   <- cumulative horizon-day return
    shifted = returns.shift(-1)
    target_df = shifted.iloc[::-1].rolling(window=horizon, min_periods=horizon).sum().iloc[::-1]

    # Features and targets are valid for different (overlapping but not
    # identical) reasons, so track them separately.
    feat_valid_mask = ~feat_df.isna().any(axis=1)
    target_valid_mask = ~target_df.isna().any(axis=1)
    valid_mask = feat_valid_mask & target_valid_mask

    n_candidate = len(feat_df)
    n_valid = int(valid_mask.sum())

    if n_valid < 100:
        raise ValueError(
            f"Not enough valid rows after feature engineering: {n_valid} "
            f"(out of {n_candidate} candidate rows). This usually means a "
            f"ticker in this universe has a much shorter listing history "
            f"than the rest, the rolling windows {windows} are too long for "
            f"the available data, or the horizon ({horizon} days) leaves too "
            f"few rows with a fully-realized future target."
        )

    valid_dates = feat_df.index[valid_mask]
    logger.info(
        f"  Feature window: {valid_dates.min().date()} → {valid_dates.max().date()} "
        f"({n_valid} usable training rows out of {n_candidate} candidate rows, "
        f"{n_candidate - n_valid} dropped to warm-up/missing-ticker/macro alignment/"
        f"horizon={horizon}d target not yet realized) "
        f"| {feat_df.shape[1]} features (returns + rolling mom/vol + macro lvl/chg)"
    )

    X = feat_df.loc[valid_mask].values
    Y = target_df.loc[valid_mask].values

    # Volume/volatility proxies for the execution-cost layer: derived from
    # raw today's-returns only (not the expanded macro/rolling feature set).
    raw_ret_valid = returns.loc[valid_mask].values
    volumes = np.abs(raw_ret_valid).mean(axis=1) * 1_000_000
    volatility = np.std(raw_ret_valid, axis=1)

    # The single most recent day with a complete FEATURE row — regardless
    # of whether its horizon-day-forward target is known yet — for live
    # forecasting. This is almost always the actual last row of the dataset
    # (features only need past data), unlike the training set's last valid
    # row, which is necessarily ~horizon days stale.
    if not feat_valid_mask.any():
        raise ValueError("No row has a complete feature vector — cannot generate a live forecast.")
    latest_idx = feat_valid_mask[feat_valid_mask].index[-1]
    latest_row = feat_df.loc[latest_idx].values
    latest_date = latest_idx

    return X, Y, volumes, volatility, tickers, latest_row, latest_date
