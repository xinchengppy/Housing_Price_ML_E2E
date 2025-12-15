"""
Feature engineering: date parts, frequency encoding, target encoding, drop leakage.

- Reads cleaned train/eval CSVs
- Applies feature engineering
- Saves feature-engineered CSVs
- ALSO saves fitted encoders for inference
- Logs progress and warnings to logs/feature_engineering.log.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump

# Configure logging
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "feature_engineering.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- feature functions ----------

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, quarter, month from date column."""
    if "date" not in df.columns:
        logger.warning("No 'date' column found; skipping date features.")
        return df
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.month
        # Place after date for readability
        df.insert(1, "year", df.pop("year"))
        df.insert(2, "quarter", df.pop("quarter"))
        df.insert(3, "month", df.pop("month"))
        logger.info("Date features added.")
    except Exception as e:
        logger.error(f"Error adding date features: {e}")
        raise
    return df


#Creates a frequency encoding (how often a value appears).
#Fit only on train, then applied to eval.
def frequency_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str):
    """Apply frequency encoding on col, fit on train."""
    if col not in train.columns or col not in eval.columns:
        logger.warning(f"Column '{col}' not in both train and eval; skipping frequency encoding.")
        return train, eval, None
    try:
        freq_map = train[col].value_counts()
        train[f"{col}_freq"] = train[col].map(freq_map)
        eval[f"{col}_freq"] = eval[col].map(freq_map).fillna(0)
        logger.info(f"Frequency encoding applied to '{col}'.")
        return train, eval, freq_map
    except Exception as e:
        logger.error(f"Error in frequency encoding for '{col}': {e}")
        raise


#Uses target encoding (replace category with average of target variable).
#Fitted only on train (prevents leakage).
def target_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str, target: str):
    """
    Use TargetEncoder on `col`, consistently name as <col>_encoded.
    For city_full → city_full_encoded (keeps schema aligned with inference).
    """
    if col not in train.columns or col not in eval.columns or target not in train.columns:
        logger.warning(f"Required columns missing for target encoding; skipping.")
        return train, eval, None
    try:
        te = TargetEncoder(cols=[col])
        encoded_col = f"{col}_encoded" if col != "city_full" else "city_full_encoded"
        train[encoded_col] = te.fit_transform(train[col], train[target])
        eval[encoded_col] = te.transform(eval[col])
        logger.info(f"Target encoding applied to '{col}'.")
        return train, eval, te
    except Exception as e:
        logger.error(f"Error in target encoding for '{col}': {e}")
        raise



def drop_unused_columns(train: pd.DataFrame, eval: pd.DataFrame):
    """Drop leakage/unused columns."""
    drop_cols = ["date", "city_full", "city", "zipcode", "median_sale_price"]
    try:
        train = train.drop(columns=[c for c in drop_cols if c in train.columns], errors="ignore")
        eval = eval.drop(columns=[c for c in drop_cols if c in eval.columns], errors="ignore")
        logger.info("Unused columns dropped.")
    except Exception as e:
        logger.error(f"Error dropping columns: {e}")
        raise
    return train, eval


# ---------- pipeline ----------

#reads cleaned CSVs → applies feature engineering → saves engineered data + encoders.
def run_feature_engineering(
    in_train_path: Path | str | None = None,
    in_eval_path: Path | str | None = None,
    in_holdout_path: Path | str | None = None,
    output_dir: Path | str = PROCESSED_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Dict], Optional[TargetEncoder]]:
    """
    Run feature engineering and write outputs + encoders to disk.
    Applies the same transformations to train, eval, and holdout.
    """
    logger.info("Starting feature engineering pipeline.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults for inputs
    if in_train_path is None:
        in_train_path = PROCESSED_DIR / "cleaning_train.csv"
    if in_eval_path is None:
        in_eval_path = PROCESSED_DIR / "cleaning_eval.csv"
    if in_holdout_path is None:
        in_holdout_path = PROCESSED_DIR / "cleaning_holdout.csv"

    try:
        train_df = pd.read_csv(in_train_path)
        eval_df = pd.read_csv(in_eval_path)
        holdout_df = pd.read_csv(in_holdout_path)
        if train_df.empty or eval_df.empty or holdout_df.empty:
            raise ValueError("One or more input DataFrames are empty")
        logger.info(f"Loaded data: Train={train_df.shape}, Eval={eval_df.shape}, Holdout={holdout_df.shape}")
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading input files: {e}")
        raise

    # Validate date ranges
    if "date" in train_df.columns:
        logger.info(f"Train date range: {train_df['date'].min()} to {train_df['date'].max()}")
    if "date" in eval_df.columns:
        logger.info(f"Eval date range: {eval_df['date'].min()} to {eval_df['date'].max()}")
    if "date" in holdout_df.columns:
        logger.info(f"Holdout date range: {holdout_df['date'].min()} to {holdout_df['date'].max()}")

    # Date features
    train_df = add_date_features(train_df)
    eval_df = add_date_features(eval_df)
    holdout_df = add_date_features(holdout_df)

    # Frequency encode zipcode (fit on train only)
    freq_map = None
    if "zipcode" in train_df.columns:
        train_df, eval_df, freq_map = frequency_encode(train_df, eval_df, "zipcode")
        holdout_df["zipcode_freq"] = holdout_df["zipcode"].map(freq_map).fillna(0)
        try:
            dump(freq_map, MODELS_DIR / "freq_encoder.pkl")
            logger.info("Frequency encoder saved.")
        except Exception as e:
            logger.error(f"Error saving frequency encoder: {e}")
            raise

    # Target encode city_full (fit on train only)
    target_encoder = None
    if "city_full" in train_df.columns:
        train_df, eval_df, target_encoder = target_encode(train_df, eval_df, "city_full", "price")
        holdout_df["city_full_encoded"] = target_encoder.transform(holdout_df["city_full"])
        try:
            dump(target_encoder, MODELS_DIR / "target_encoder.pkl")
            logger.info("Target encoder saved.")
        except Exception as e:
            logger.error(f"Error saving target encoder: {e}")
            raise

    # Drop leakage / raw categoricals
    train_df, eval_df = drop_unused_columns(train_df, eval_df)
    holdout_df, _ = drop_unused_columns(holdout_df.copy(), holdout_df.copy())

    # Save engineered data
    splits = [
        ("train", train_df, output_dir / "feature_engineered_train.csv"),
        ("eval", eval_df, output_dir / "feature_engineered_eval.csv"),
        ("holdout", holdout_df, output_dir / "feature_engineered_holdout.csv"),
    ]
    for name, df, path in splits:
        try:
            df.to_csv(path, index=False)
            logger.info(f"Saved {name} to {path}")
        except Exception as e:
            logger.error(f"Error saving {name} to {path}: {e}")
            raise

    logger.info("Feature engineering complete.")
    return train_df, eval_df, holdout_df, freq_map, target_encoder


if __name__ == "__main__":
    try:
        run_feature_engineering()
        logger.info("Feature engineering completed successfully.")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        sys.exit(1)
