"""Data cleaning and sanitization for Telco Customer Churn."""

import pandas as pd
import numpy as np


class DataCleaner:
    """Sanitizes raw Telco churn tabular data."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Handle blank spaces in TotalCharges
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
            # If tenure == 0, TotalCharges is typically 0
            df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

        # Standardize target Churn column if present
        if "Churn" in df.columns and df["Churn"].dtype == object:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

        # Drop customerID if present
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        return df
