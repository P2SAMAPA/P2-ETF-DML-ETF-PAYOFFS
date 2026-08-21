"""
trainer.py  —  Differential Machine Learning Trainer

FIXES vs original:
  1. The model previously had output_dim=1 and was trained on a
     cross-sectional MEAN next-day return (one number per day). But
     generate_picks() then zipped that single-day scalar list against the
     list of ticker symbols — i.e. it was labeling a market-wide daily
     forecast as if it were N different per-ticker forecasts. Now the
     model has output_dim = n_tickers and is trained directly on a
     per-ticker target matrix (see data_manager.prepare_features), so each
     output neuron corresponds to one ticker and "top picks" are actually
     comparing different tickers' forecasts against each other.
  2. Feature normalization is now fit on the TRAIN split only and applied
     to the validation split with those same stats (previously the whole
     X array — train and validation together — was normalized before the
     split, leaking validation-set statistics into training).
  3. generate_picks() now predicts from the single most recent feature row
     (i.e. an actual forecast for the next trading day) instead of
     mismatching 50 historical rows against a ticker list of a different
     length.
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data, prepare_features, filter_by_history
from differential_layers import DifferentialNetwork
from execution_models import ExecutionCostCalculator
from payoff_functions import DiscontinuityAwareLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_picks(model, x_latest_row: np.ndarray, tickers: List[str],
                   ticker_vol: np.ndarray, top_n: int = 1,
                   horizon_days: int = 1) -> List[Dict]:
    """
    Generate top N ETF picks from the model's forecast for the given
    feature row (should be the TRUE most recent trading day's features —
    see trainer.py's run_trainer for why that's not simply the training
    set's last row once horizon > 1).

    ticker_vol: (n_tickers,) each ticker's own recent realized volatility,
    matched to the same horizon as the prediction itself (i.e. std of the
    actual horizon-day-forward return series, not daily vol). Used to turn
    each raw prediction into a z-score — "how many standard deviations of
    this ticker's normal `horizon_days`-day move is the model forecasting"
    — rather than comparing the raw prediction against a fixed absolute cutoff.

    Confidence is based on |z| = |prediction| / ticker's own recent
    (horizon-matched) volatility: High >= 2sigma, Medium >= 1sigma, else Low.
    Symmetric in sign, so a large negative forecast is correctly flagged as
    high-confidence too, not lumped in with negligible predictions.
    """
    device = next(model.parameters()).device

    X_t = torch.FloatTensor(x_latest_row).reshape(1, -1).to(device)

    model.eval()
    with torch.no_grad():
        predictions, _ = model(X_t, None, None)

    preds = predictions.cpu().numpy().flatten()  # (n_tickers,)

    if len(preds) != len(tickers):
        raise ValueError(
            f"Prediction length ({len(preds)}) does not match ticker count "
            f"({len(tickers)}) — model output_dim must equal n_tickers."
        )
    if len(ticker_vol) != len(tickers):
        raise ValueError(
            f"ticker_vol length ({len(ticker_vol)}) does not match ticker count "
            f"({len(tickers)})."
        )

    ticker_preds = list(zip(tickers, preds, ticker_vol))
    sorted_picks = sorted(ticker_preds, key=lambda x: x[1], reverse=True)
    top_picks = sorted_picks[:top_n]

    horizon_label = f"{horizon_days}d" if horizon_days != 1 else "1d"

    results = []
    for ticker, pred, vol in top_picks:
        z = float(pred) / (float(vol) + 1e-8)
        az = abs(z)
        confidence = "High" if az >= 2.0 else "Medium" if az >= 1.0 else "Low"
        results.append({
            "ticker": ticker,
            "horizon_days": horizon_days,
            "expected_return": round(float(pred) * 100, 2),  # Convert to percentage
            "confidence": confidence,
            "z_score": round(z, 2),
            "rationale": (
                f"DML {horizon_label} forward prediction: {float(pred):.4f} "
                f"({z:+.2f}\u03c3 vs {ticker}'s recent {horizon_label} realized vol of {vol*100:.2f}%)"
            )
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

    # Initialize execution cost calculator (used for reporting/diagnostics)
    exec_calc = ExecutionCostCalculator(config.EXECUTION_CONFIG)

    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Training on {universe_name}...")

        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            logger.warning(f"No tickers available for {universe_name}")
            continue

        # Exclude recently-listed tickers with insufficient history so they
        # don't force the whole universe's usable training window down to
        # their inception date (e.g. a universe with mostly 2004+ tickers
        # was previously capped at ~6.5 years by a single 2018-listed ETF).
        min_days = config.MIN_HISTORY_DAYS
        available, excluded = filter_by_history(prices_df, available, min_days)

        if excluded:
            logger.info(
                f"  Excluding {len(excluded)} recently-listed ticker(s) with < {min_days} "
                f"trading days of history: " +
                ", ".join(f"{t} ({n} days)" for t, n in excluded.items())
            )

        if len(available) < 2:
            logger.warning(
                f"Not enough long-history tickers left for {universe_name} "
                f"({len(available)} remain after excluding short-history tickers) — skipping"
            )
            continue

        # Prepare features/targets: X is (n_days, n_tickers), Y is
        # (n_days, n_tickers) — one target per ticker, per day, now the
        # cumulative return over the next PREDICTION_HORIZON_DAYS days.
        # latest_row/latest_date are the TRUE most recent day's features —
        # NOT part of the training set, since with horizon > 1 the training
        # set's last row is necessarily ~horizon days stale (its target
        # can't be known until `horizon` future days have actually happened).
        try:
            X, Y, volumes, volatilities, feature_tickers, latest_row_raw, latest_date = prepare_features(
                prices_df[available], macro_df, horizon=config.PREDICTION_HORIZON_DAYS
            )
        except Exception as e:
            logger.error(f"Failed to prepare features for {universe_name}: {e}")
            continue

        logger.info(f"Feature shape: {X.shape}, Target shape: {Y.shape}")

        if len(X) != len(Y):
            logger.error(f"Shape mismatch: X={X.shape}, Y={Y.shape}")
            continue

        if len(X) < 100:
            logger.warning(f"Not enough data for {universe_name}: {len(X)} samples")
            continue

        # IMPORTANT: n_tickers (the model's output_dim) must come from Y's
        # column count, NOT X's. X now includes macro + rolling-window
        # features in addition to per-ticker returns, so X.shape[1] is no
        # longer equal to the number of tickers being predicted.
        n_tickers = Y.shape[1]

        # Split into train and validation FIRST, then normalize using only
        # the train split's statistics (avoids leaking validation info).
        #
        # PURGE GAP: with horizon > 1, row t's target spans days t+1..t+horizon,
        # so adjacent rows' targets overlap almost entirely (e.g. horizon=21:
        # adjacent rows share 20 of 21 days -- verified directly on synthetic
        # pure-noise data: the last training row's target and the first
        # validation row's target were correlated at 0.97 with a naive split,
        # dropping to -0.03 (pure noise, as it should be) once this gap is
        # added). Without the gap, the model can partially "match" validation
        # targets via near-duplicate training targets right at the boundary
        # rather than genuinely generalizing -- inflating early validation
        # performance and making a very early epoch look artificially best.
        split_idx = int(len(X) * 0.8)
        purge = config.PREDICTION_HORIZON_DAYS
        train_end = max(split_idx - purge, int(len(X) * 0.5))  # don't purge away most of training on tiny datasets

        X_train_raw, X_val_raw = X[:train_end], X[split_idx:]
        Y_train, Y_val = Y[:train_end], Y[split_idx:]
        V_train, V_val = volumes[:train_end], volumes[split_idx:]
        Vol_train, Vol_val = volatilities[:train_end], volatilities[split_idx:]

        if train_end < split_idx:
            logger.info(f"  Purge gap: {split_idx - train_end} rows excluded between train and validation "
                        f"(horizon={config.PREDICTION_HORIZON_DAYS}d, prevents overlapping-target leakage)")

        feat_mean = X_train_raw.mean(axis=0)
        feat_std = X_train_raw.std(axis=0) + 1e-8

        X_train = (X_train_raw - feat_mean) / feat_std
        X_val = (X_val_raw - feat_mean) / feat_std

        # Convert to tensors
        X_t = torch.FloatTensor(X_train).to(device)
        Y_t = torch.FloatTensor(Y_train).to(device)  # (n, n_tickers)
        V_t = torch.FloatTensor(V_train).reshape(-1, 1).to(device)
        Vol_t = torch.FloatTensor(Vol_train).reshape(-1, 1).to(device)

        # Validation tensors
        X_val_t = torch.FloatTensor(X_val).to(device)
        Y_val_t = torch.FloatTensor(Y_val).to(device)
        V_val_t = torch.FloatTensor(V_val).reshape(-1, 1).to(device)
        Vol_val_t = torch.FloatTensor(Vol_val).reshape(-1, 1).to(device)

        # Create dataset
        dataset = TensorDataset(X_t, Y_t, V_t, Vol_t)
        dataloader = DataLoader(dataset, batch_size=config.DML_CONFIG["batch_size"], shuffle=True)

        # Initialize model — output_dim = n_tickers so each output neuron
        # is a per-ticker next-day return forecast.
        model = DifferentialNetwork(
            input_dim=X.shape[1],
            output_dim=n_tickers,
            config=config.DML_CONFIG
        ).to(device)

        # Optimizer. weight_decay comes from config.DML_CONFIG["l2_reg"] — this
        # was already defined in config.py but never actually applied anywhere,
        # meaning the model had zero L2 regularization despite the config
        # implying otherwise. With ~330 features and only ~1600 samples for
        # the larger universes, this is very likely why validation loss
        # peaked within the first 2-3 epochs in practice.
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.DML_CONFIG["learning_rate"],
            weight_decay=config.DML_CONFIG.get("l2_reg", 0.0)
        )

        # Training loop
        logger.info(f"  Training {len(dataloader)} batches for {config.DML_CONFIG['epochs']} epochs...")
        model.train()

        best_val_mse = float('inf')
        avg_mse = float('inf')
        best_model_state = None
        best_epoch = 0
        best_train_mse = float('inf')

        for epoch in range(config.DML_CONFIG["epochs"]):
            epoch_mse = 0.0
            epoch_diff = 0.0
            n_batches_seen = 0

            for X_batch, Y_batch, V_batch, Vol_batch in dataloader:
                optimizer.zero_grad()

                # prediction is ALWAYS the raw return forecast now (never
                # cost-adjusted) — see differential_layers.py's forward().
                prediction, info = model(X_batch, V_batch, Vol_batch)

                # The actual "differential loss": MSE on the raw prediction
                # PLUS a regularizer that penalizes high execution-cost
                # SENSITIVITY. This was previously defined but never called.
                loss_dict = model.compute_differential_loss(
                    prediction, Y_batch, V_batch, Vol_batch,
                    diff_loss_weight=config.DML_CONFIG.get("differential_weight", 1e-4)
                )
                loss = loss_dict["total_loss"]

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"    Non-finite loss encountered at epoch {epoch} — skipping this batch's step")
                    continue

                loss.backward()
                # Defensive: clip gradients so a bad batch/parameter regime
                # can't blow up weights in one step (belt-and-suspenders on
                # top of the bounded impact/gamma fix in differential_layers.py).
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_mse += loss_dict["mse_loss"].item()
                epoch_diff += loss_dict["diff_loss"].item()
                n_batches_seen += 1

            avg_mse = epoch_mse / max(n_batches_seen, 1)
            avg_diff = epoch_diff / max(n_batches_seen, 1)

            # Validation — same raw prediction pathway as training, so
            # train and validation are directly comparable (no more
            # cost-adjusted-vs-raw mismatch).
            model.eval()
            with torch.no_grad():
                val_pred, _ = model(X_val_t, V_val_t, Vol_val_t)
                val_loss_dict = model.compute_differential_loss(
                    val_pred, Y_val_t, V_val_t, Vol_val_t,
                    diff_loss_weight=config.DML_CONFIG.get("differential_weight", 1e-4)
                )
                val_mse = val_loss_dict["mse_loss"].item()
            model.train()

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
                best_train_mse = avg_mse  # train MSE AT the best epoch — directly comparable to best_val_mse

            if epoch % 20 == 0:
                logger.info(
                    f"    Epoch {epoch}: Train MSE={avg_mse:.6f} (+diff_loss={avg_diff:.6f}), "
                    f"Val MSE={val_mse:.6f}"
                )

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Generate picks from the TRUE most recent trading day's features
        # (normalized with the same train-split stats used everywhere else)
        # — an actual live forecast, not one that's ~horizon days stale.
        #
        # ticker_vol: each ticker's own recent realized volatility OF THE
        # SAME HORIZON as the target itself (Y is already the cumulative
        # horizon-day return, so std(Y) directly gives horizon-matched
        # volatility — no separate daily-vol-times-sqrt(horizon) scaling
        # needed). The window is widened for longer horizons since Y's rows
        # overlap heavily (adjacent horizon-day windows share almost all
        # their days), so more rows are needed for a stable estimate.
        vol_window = min(max(63, config.PREDICTION_HORIZON_DAYS * 12), len(Y))
        ticker_vol = Y[-vol_window:].std(axis=0)

        latest_row = (latest_row_raw - feat_mean) / feat_std
        logger.info(f"  Live forecast as of: {latest_date.date()} (horizon: {config.PREDICTION_HORIZON_DAYS}d forward)")
        picks = generate_picks(model, latest_row, feature_tickers, ticker_vol,
                               top_n=config.TOP_N_PICKS, horizon_days=config.PREDICTION_HORIZON_DAYS)

        # Store results — train_mse is now reported AT the best (saved)
        # epoch, so it's directly comparable to val_mse (previously "loss"
        # was the LAST epoch's train loss, which could differ from the
        # epoch actually used for the saved model/picks).
        results["universes"][universe_name] = {
            "train_mse_at_best_epoch": best_train_mse,
            "best_val_mse": best_val_mse,
            "best_epoch": best_epoch,
            "final_epoch_train_mse": avg_mse,
            "tickers": available,
            "excluded_short_history": excluded,
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
