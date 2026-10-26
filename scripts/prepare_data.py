"""Prepares raw Telco Churn dataset, partitions 80/20 train-val & sealed holdout, and generates manifest."""

import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cleaner import DataCleaner
from src.data.splitter import create_stratified_partitions
from src.data.validator import DataProvenanceGuard
from scripts.run_autonomous_loop import generate_sample_tabular_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DataPreparer")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset partitions and cryptographic manifest.")
    parser.add_argument("--raw-path", type=str, default="data/raw/Telco-Customer-Churn.csv", help="Path to raw CSV")
    parser.add_argument("--force-sample", action="store_true", help="Generate synthetic sample data if raw CSV is absent")
    args = parser.parse_args()

    raw_file = Path(args.raw_path)
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not raw_file.exists():
        logger.warning("Raw dataset not found at %s.", raw_file)
        logger.info("Downloading official Telco Customer Churn dataset (blastchar / IBM)...")
        import urllib.request
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        urllib.request.urlretrieve(url, raw_file)
        logger.info("Successfully downloaded dataset to %s", raw_file)
        raw_df = pd.read_csv(raw_file)
    else:
        logger.info("Loading raw dataset from %s...", raw_file)
        raw_df = pd.read_csv(raw_file)

    logger.info("Raw dataset shape: %s", raw_df.shape)

    # 1. Cleaning & sanitization
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean(raw_df)

    # 2. Stratified 80/20 train-val and sealed holdout split
    train_val_df, holdout_df = create_stratified_partitions(cleaned_df, target_col="Churn", holdout_ratio=0.20)
    logger.info("Partitioned: Train-Val=%d rows, Sealed Holdout=%d rows", len(train_val_df), len(holdout_df))

    # 3. Save Parquet partitions
    train_val_path = processed_dir / "train_val.parquet"
    holdout_path = processed_dir / "sealed_holdout.parquet"
    train_val_df.to_parquet(train_val_path, index=False)
    holdout_df.to_parquet(holdout_path, index=False)
    logger.info("Saved partitions to %s and %s", train_val_path, holdout_path)

    # 4. Generate Cryptographic Manifest
    manifest_path = artifacts_dir / "dataset_manifest.json"
    raw_hash = DataProvenanceGuard.compute_file_sha256(raw_file)
    train_hash = DataProvenanceGuard.compute_file_sha256(train_val_path)
    content_hash = DataProvenanceGuard.compute_dataframe_content_hash(train_val_df)

    manifest_data = {
        "raw_file_sha256": raw_hash,
        "canonical_train_sha256": train_hash,
        "content_row_hash": content_hash,
        "total_rows": len(train_val_df),
        "allowed_indices": list(train_val_df.index),
        "allowed_columns": list(train_val_df.columns),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info("Cryptographic manifest successfully created at %s", manifest_path)
    logger.info("Data preparation completed with zero experiment budget cost.")


if __name__ == "__main__":
    main()
