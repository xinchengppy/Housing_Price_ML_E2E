"""
Load & time-split the raw dataset.

- Production default writes to data/raw/
- Tests can pass a temp `output_dir` so nothing in data/ is touched.
- Logs progress and warnings to logs/load.log.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

# Configure logging
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "load.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")

# Constants
CUTOFF_EVAL = pd.Timestamp("2020-01-01")  # eval starts
CUTOFF_HOLDOUT = pd.Timestamp("2022-01-01")  # holdout starts


def load_and_split_data(
    raw_path: str = "data/raw/untouched_raw_original.csv",
    output_dir: Path | str = DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw dataset, split into train/eval/holdout by date, and save to output_dir."""
    logger.info(f"Starting data loading and splitting from {raw_path}")

    try:
        df = pd.read_csv(raw_path)
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")
        logger.info(f"Loaded raw data: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Raw data file not found: {raw_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Raw data file is empty: {raw_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
        raise

    # Validate required columns
    required_cols = {"date"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    try:
        # Ensure datetime + sort
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isnull().any():
            logger.warning("Some date values could not be parsed; they will be NaT")
        df = df.sort_values("date")
        logger.info("Data sorted by date.")
    except Exception as e:
        logger.error(f"Error processing dates: {e}")
        raise

    # Cutoffs
    cutoff_date_eval = CUTOFF_EVAL
    cutoff_date_holdout = CUTOFF_HOLDOUT
    logger.info(f"Using cutoffs: eval={cutoff_date_eval}, holdout={cutoff_date_holdout}")

    # Splits
    train_df = df[df["date"] < cutoff_date_eval]
    eval_df = df[(df["date"] >= cutoff_date_eval) & (df["date"] < cutoff_date_holdout)]
    holdout_df = df[df["date"] >= cutoff_date_holdout]

    logger.info(f"Splits created: Train={train_df.shape}, Eval={eval_df.shape}, Holdout={holdout_df.shape}")

    # Validate splits
    if train_df.empty:
        logger.warning("Train split is empty")
    if eval_df.empty:
        logger.warning("Eval split is empty")
    if holdout_df.empty:
        logger.warning("Holdout split is empty")

    # Save
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits = [
        ("train", train_df),
        ("eval", eval_df),
        ("holdout", holdout_df),
    ]
    for name, split_df in splits:
        out_path = outdir / f"{name}.csv"
        try:
            split_df.to_csv(out_path, index=False)
            logger.info(f"Saved {name} to {out_path}")
        except Exception as e:
            logger.error(f"Error saving {name} to {out_path}: {e}")
            raise

    logger.info(f"Data split completed (saved to {outdir}).")
    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    try:
        load_and_split_data()
        logger.info("Load and split completed successfully.")
    except Exception as e:
        logger.error(f"Load and split failed: {e}")
        sys.exit(1)
