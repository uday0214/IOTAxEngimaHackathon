"""Data loading and canonical path management."""

from pathlib import Path
from typing import Optional
import pandas as pd


class DataLoader:
    """Manages secure reading of raw and partitioned Telco datasets."""

    def __init__(self, raw_path: Path = Path("data/raw/Telco-Customer-Churn.csv")):
        self.raw_path = Path(raw_path)

    def load_raw(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw dataset file not found at: {self.raw_path}")
        return pd.read_csv(self.raw_path)

    def load_train_val(self, path: Path = Path("data/processed/train_val.parquet")) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Processed train_val dataset not found at: {path}")
        return pd.read_parquet(path)

    def load_holdout(self, path: Path = Path("data/processed/sealed_holdout.parquet")) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Sealed holdout dataset not found at: {path}")
        return pd.read_parquet(path)
