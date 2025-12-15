from __future__ import annotations
"""
Evaluate a saved XGBoost model on the eval split.
"""

import logging
import sys

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "eval.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_EVAL = Path("data/processed/feature_engineered_eval.csv")
DEFAULT_MODEL = Path("models/xgb_model.pkl")


def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    """Optionally sample DataFrame for faster evaluation/testing."""
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


def evaluate_model(
    model_path: Path | str = DEFAULT_MODEL,
    eval_path: Path | str = DEFAULT_EVAL,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate saved model on eval data and return metrics."""
    logger.info("Starting model evaluation.")

    try:
        eval_df = pd.read_csv(eval_path)
        if eval_df.empty:
            raise ValueError("Eval DataFrame is empty")
        logger.info(f"Loaded eval data: {eval_df.shape}")
    except FileNotFoundError:
        logger.error(f"Eval data file not found: {eval_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading eval data: {e}")
        raise

    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    target = "price"
    if target not in eval_df.columns:
        raise ValueError(f"Target column '{target}' missing in eval data")

    try:
        X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]
        logger.info(f"Split eval data: X_eval={X_eval.shape}, y_eval={len(y_eval)}")
    except Exception as e:
        logger.error(f"Error splitting features/target: {e}")
        raise

    try:
        model = load(model_path)
        logger.info(f"Loaded model from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    try:
        y_pred = model.predict(X_eval)
        mae = float(mean_absolute_error(y_eval, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
        r2 = float(r2_score(y_eval, y_pred))
        metrics = {"mae": mae, "rmse": rmse, "r2": r2}
        logger.info(f"Evaluation metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

    return metrics


if __name__ == "__main__":
    try:
        metrics = evaluate_model()
        logger.info("Evaluation completed successfully.")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)
