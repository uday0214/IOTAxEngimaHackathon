"""Dataset splitting: 80% Train-Val (active) and 20% Sealed Holdout (locked)."""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def create_stratified_partitions(
    df: pd.DataFrame,
    target_col: str = "Churn",
    holdout_ratio: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partitions the dataset into 80% Train-Val pool and 20% Sealed Holdout.
    
    Zero experiment budget cost.
    """
    stratify = df[target_col] if target_col in df.columns else None
    train_val_df, holdout_df = train_test_split(
        df,
        test_size=holdout_ratio,
        random_state=random_state,
        stratify=stratify,
    )
    return train_val_df.reset_index(drop=True), holdout_df.reset_index(drop=True)
