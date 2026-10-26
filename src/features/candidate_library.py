"""Domain feature candidate transformations for Telco Customer Churn."""

import pandas as pd
import numpy as np


class ChurnFeatureLibrary:
    """Computes leak-free domain features for customer churn prediction."""

    @staticmethod
    def add_charge_ratios(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
            df["MonthlyToTotalRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1.0)
            if "tenure" in df.columns:
                df["ExpectedTotalDiff"] = (df["MonthlyCharges"] * df["tenure"]) - df["TotalCharges"]
        return df

    @staticmethod
    def add_tenure_cohorts(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "tenure" in df.columns:
            df["TenureGroup"] = pd.cut(
                df["tenure"],
                bins=[-1, 12, 24, 48, 72],
                labels=["0-12m", "12-24m", "24-48m", "48-72m"],
            ).astype(str)
            df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)
        return df

    @staticmethod
    def add_service_density(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        service_cols = [
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        active_services = [col for col in service_cols if col in df.columns]
        if active_services:
            df["TotalServices"] = (df[active_services] == "Yes").sum(axis=1)

        streaming_cols = [col for col in ["StreamingTV", "StreamingMovies"] if col in df.columns]
        if streaming_cols and "TotalServices" in df.columns:
            df["StreamingRatio"] = (df[streaming_cols] == "Yes").sum(axis=1) / (df["TotalServices"] + 1e-4)
        return df

    @staticmethod
    def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Contract" in df.columns and "PaymentMethod" in df.columns:
            df["Contract_x_Payment"] = df["Contract"].astype(str) + "__" + df["PaymentMethod"].astype(str)
        if "Contract" in df.columns and "InternetService" in df.columns:
            df["Contract_x_Internet"] = df["Contract"].astype(str) + "__" + df["InternetService"].astype(str)
        return df
