"""
Preprocessing Script for Housing Regression MLE

- Reads train/eval/holdout CSVs from data/raw/.
- Cleans and normalizes city names.
- Maps cities to metros and merges lat/lng.
- Drops duplicates and extreme outliers.
- Saves cleaned splits to data/processed/.
- Logs progress and warnings to logs/preprocess.log.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

# Configure logging
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "preprocess.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# Constants
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTLIER_THRESHOLD = 19_000_000  # $19M

# Manual fixes for known mismatches (normalized form)
CITY_MAPPING = {
    "las vegas-henderson-paradise": "las vegas-henderson-north las vegas",
    "denver-aurora-lakewood": "denver-aurora-centennial",
    "houston-the woodlands-sugar land": "houston-pasadena-the woodlands",
    "austin-round rock-georgetown": "austin-round rock-san marcos",
    "miami-fort lauderdale-pompano beach": "miami-fort lauderdale-west palm beach",
    "san francisco-oakland-berkeley": "san francisco-oakland-fremont",
    "dc_metro": "washington-arlington-alexandria",
    "atlanta-sandy springs-alpharetta": "atlanta-sandy springs-roswell",
}


def normalize_city(s: str) -> str:
    """Lowercase, strip, unify dashes. Safe for NA."""
    if pd.isna(s):
        return s
    try:
        s = str(s).strip().lower()
        s = re.sub(r"[–—-]", "-", s)  # unify dashes
        s = re.sub(r"\s+", " ", s)  # collapse spaces
        return s
    except Exception as e:
        logger.warning(f"Error normalizing city '{s}': {e}")
        return s


def clean_and_merge(df: pd.DataFrame, metros_path: Optional[str] = "data/raw/usmetros.csv") -> pd.DataFrame:
    """
    Normalize city names, optionally merge lat/lng from metros dataset.
    If `city_full` column or `metros_path` is missing, skip gracefully.
    """
    if "city_full" not in df.columns:
        logger.info("Skipping city merge: no 'city_full' column present.")
        return df

    try:
        # Normalize city_full
        df["city_full"] = df["city_full"].apply(normalize_city)
        # Apply mapping
        norm_mapping = {normalize_city(k): normalize_city(v) for k, v in CITY_MAPPING.items()}
        df["city_full"] = df["city_full"].replace(norm_mapping)
        logger.info("City normalization and mapping applied.")
    except Exception as e:
        logger.error(f"Error during city normalization: {e}")
        raise

    # If lat/lng already present, skip merge
    if {"lat", "lng"}.issubset(df.columns):
        logger.info("Skipping lat/lng merge: already present in DataFrame.")
        return df

    # If no metros file provided / exists, skip merge
    if not metros_path or not Path(metros_path).exists():
        logger.info("Skipping lat/lng merge: metros file not provided or not found.")
        return df

    try:
        # Merge lat/lng
        metros = pd.read_csv(metros_path)
        required_cols = {"metro_full", "lat", "lng"}
        if not required_cols.issubset(metros.columns):
            logger.warning("Skipping lat/lng merge: metros file missing required columns.")
            return df

        metros["metro_full"] = metros["metro_full"].str.split(",").str[0]
        metros["metro_full"] = metros["metro_full"].apply(normalize_city)

        df = df.merge(
            metros[["metro_full", "lat", "lng"]],
            how="left",
            left_on="city_full",
            right_on="metro_full",
        )
        df.drop(columns=["metro_full"], inplace=True, errors="ignore")

        missing = df[df["lat"].isnull()]["city_full"].unique()
        if len(missing) > 0:
            logger.warning(f"Still missing lat/lng for: {missing}")
        else:
            logger.info("All cities matched with metros dataset.")
    except FileNotFoundError:
        logger.error(f"Metros file not found: {metros_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Metros file is empty: {metros_path}")
        raise
    except Exception as e:
        logger.error(f"Error during lat/lng merge: {e}")
        raise

    return df



def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates while keeping different dates/years."""
    try:
        before = df.shape[0]
        df = df.drop_duplicates(subset=df.columns.difference(["date", "year"]), keep=False)
        after = df.shape[0]
        logger.info(f"Dropped {before - after} duplicate rows (excluding date/year).")
    except Exception as e:
        logger.error(f"Error dropping duplicates: {e}")
        raise
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers in median_list_price (> OUTLIER_THRESHOLD)."""
    if "median_list_price" not in df.columns:
        logger.info("No median_list_price column found; skipping outlier removal.")
        return df
    try:
        before = df.shape[0]
        df = df[df["median_list_price"] <= OUTLIER_THRESHOLD].copy()
        after = df.shape[0]
        logger.info(f"Removed {before - after} rows with median_list_price > {OUTLIER_THRESHOLD}.")
    except Exception as e:
        logger.error(f"Error removing outliers: {e}")
        raise
    return df


def preprocess_split(
    split: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: Optional[str] = "data/raw/usmetros.csv",
) -> pd.DataFrame:
    """Run preprocessing for a split and save to processed_dir."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}.csv"
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Loaded DataFrame is empty for {split}")
        logger.info(f"Loaded {split} data: {df.shape}")
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        raise

    df = clean_and_merge(df, metros_path=metros_path)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    out_path = processed_dir / f"cleaning_{split}.csv"
    try:
        df.to_csv(out_path, index=False)
        logger.info(f"Preprocessed {split} saved to {out_path} ({df.shape})")
    except Exception as e:
        logger.error(f"Error saving to {out_path}: {e}")
        raise
    return df


def run_preprocess(
    splits: tuple[str, ...] = ("train", "eval", "holdout"),
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: Optional[str] = "data/raw/usmetros.csv",
):
    """Run preprocessing for all splits."""
    logger.info("Starting preprocessing pipeline.")
    try:
        for s in splits:
            preprocess_split(s, raw_dir=raw_dir, processed_dir=processed_dir, metros_path=metros_path)
        logger.info("Preprocessing pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_preprocess()
