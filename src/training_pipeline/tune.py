from __future__ import annotations
"""
Hyperparameter tuning with Optuna + MLflow.

- Optimizes XGB params on eval set RMSE.
- Logs trials to MLflow.
- Retrains best model and saves to `model_output`.
"""

import logging
import sys

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Configure logging
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "tune.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_TRAIN = Path("data/processed/feature_engineered_train.csv")
DEFAULT_EVAL = Path("data/processed/feature_engineered_eval.csv")
DEFAULT_OUT = Path("models/xgb_best_model.pkl")


def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    """Optionally sample DataFrame for faster tuning/testing."""
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


def _load_data(
    train_path: Path | str,
    eval_path: Path | str,
    sample_frac: Optional[float],
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and split data."""
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

    return X_train, y_train, X_eval, y_eval


def tune_model(
    train_path: Path | str = DEFAULT_TRAIN,
    eval_path: Path | str = DEFAULT_EVAL,
    model_output: Path | str = DEFAULT_OUT,
    n_trials: int = 15,
    sample_frac: Optional[float] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "xgboost_optuna_housing",
    random_state: int = 42,
) -> Tuple[Dict, Dict]:
    """Run Optuna tuning; save best model; return (best_params, best_metrics)."""
    logger.info("Starting hyperparameter tuning with Optuna + MLflow.")

    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to '{experiment_name}'")
    except Exception as e:
        logger.error(f"Error setting up MLflow: {e}")
        raise

    X_train, y_train, X_eval, y_eval = _load_data(train_path, eval_path, sample_frac, random_state)

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        try:
            with mlflow.start_run(nested=True):
                model = XGBRegressor(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_eval)
                rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
                mae = float(mean_absolute_error(y_eval, y_pred))
                r2 = float(r2_score(y_eval, y_pred))

                mlflow.log_params(params)
                mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            raise optuna.TrialPruned()

        return rmse

    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        logger.info(f"Optuna optimization completed with {n_trials} trials.")
    except Exception as e:
        logger.error(f"Error during Optuna optimization: {e}")
        raise

    best_params = study.best_trial.params
    logger.info(f"Best params from Optuna: {best_params}")

    # Retrain best model
    try:
        best_model = XGBRegressor(**{**best_params, "random_state": random_state, "n_jobs": -1, "tree_method": "hist"})
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_eval)
        best_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_eval, y_pred))),
            "mae": float(mean_absolute_error(y_eval, y_pred)),
            "r2": float(r2_score(y_eval, y_pred)),
        }
        logger.info(f"Best tuned model metrics: {best_metrics}")
    except Exception as e:
        logger.error(f"Error retraining best model: {e}")
        raise

    # Save to models/
    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        dump(best_model, out)
        logger.info(f"Best model saved to {out}")
    except Exception as e:
        logger.error(f"Error saving best model to {out}: {e}")
        raise

    # Log final best model to MLflow
    try:
        with mlflow.start_run(run_name="best_xgb_model"):
            mlflow.log_params(best_params)
            mlflow.log_metrics(best_metrics)
            mlflow.sklearn.log_model(best_model, "model")
        logger.info("Best model logged to MLflow.")
    except Exception as e:
        logger.error(f"Error logging to MLflow: {e}")
        raise

    return best_params, best_metrics


if __name__ == "__main__":
    try:
        tune_model()
        logger.info("Tuning completed successfully.")
    except Exception as e:
        logger.error(f"Tuning failed: {e}")
        sys.exit(1)
