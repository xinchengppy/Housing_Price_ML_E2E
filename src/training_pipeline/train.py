from __future__ import annotations

"""
Train a baseline XGBoost model.

- Reads feature-engineered train/eval CSVs.
- Trains XGBRegressor.
- Returns metrics and saves model to `model_output`.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Configure logging
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "train.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_TRAIN = Path("data/processed/feature_engineered_train.csv")
DEFAULT_EVAL = Path("data/processed/feature_engineered_eval.csv")
DEFAULT_OUT = Path("models/xgb_model.pkl")


def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    """Optionally sample DataFrame for faster training/testing."""
    if sample_frac is None:
        return df
    try:
        sample_frac = float(sample_frac)
        if sample_frac <= 0 or sample_frac >= 1:
            logger.info("Sample fraction out of range (0,1); using full data.")
            return df
        sampled = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
        logger.info(f"Sampled {len(sampled)} rows from {len(df)} (frac={sample_frac}).")
        return sampled
    except Exception as e:
        logger.error(f"Error sampling data: {e}")
        raise


def train_model(
    train_path: Path | str = DEFAULT_TRAIN,
    eval_path: Path | str = DEFAULT_EVAL,
    model_output: Path | str = DEFAULT_OUT,
    model_params: Optional[Dict] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[XGBRegressor, Dict[str, float]]:
    """Train baseline XGB and save model.

    Returns
    -------
    model : XGBRegressor
    metrics : dict[str, float]
    """
    logger.info("Starting model training.")

    try:
        train_df = pd.read_csv(train_path)
        eval_df = pd.read_csv(eval_path)
        if train_df.empty or eval_df.empty:
            raise ValueError("Train or eval DataFrame is empty")
        logger.info(f"Loaded train: {train_df.shape}, eval: {eval_df.shape}")
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    train_df = _maybe_sample(train_df, sample_frac, random_state)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    target = "price"
    if target not in train_df.columns or target not in eval_df.columns:
        raise ValueError(f"Target column '{target}' missing in train or eval data")

    try:
        X_train, y_train = train_df.drop(columns=[target]), train_df[target]
        X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]
        logger.info(f"Split data: X_train={X_train.shape}, y_train={len(y_train)}")
    except Exception as e:
        logger.error(f"Error splitting features/target: {e}")
        raise

    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if model_params:
        params.update(model_params)
        logger.info(f"Updated params: {model_params}")

    try:
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

    try:
        y_pred = model.predict(X_eval)
        mae = float(mean_absolute_error(y_eval, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
        r2 = float(r2_score(y_eval, y_pred))
        metrics = {"mae": mae, "rmse": rmse, "r2": r2}
        logger.info(f"Evaluation metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        dump(model, out)
        logger.info(f"Model saved to {out}")
    except Exception as e:
        logger.error(f"Error saving model to {out}: {e}")
        raise

    return model, metrics


if __name__ == "__main__":
    try:
        train_model()
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
