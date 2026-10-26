"""Pre-Flight Dataset Profiler & Classifier Selector (Step 0, Zero Experiment Budget Cost).

Extracts structural meta-features and evaluates inductive bias to lock the single best classifier.
"""

from dataclasses import dataclass
import logging
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetaFeatures:
    num_samples: int
    num_features: int
    categorical_count: int
    numerical_count: int
    categorical_ratio: float
    class_imbalance_ratio: float
    max_cardinality: int
    missing_cell_ratio: float


@dataclass
class ClassifierLockDecision:
    selected_family: str  # "catboost"
    rationale: str
    inductive_bias_fit_score: float
    is_locked: bool = True


class PreFlightProfiler:
    """Profiles tabular dataset properties and determines the optimal classifier."""

    def extract_meta_features(self, df: pd.DataFrame, target_col: str = "Churn") -> DatasetMetaFeatures:
        features_df = df.drop(columns=[target_col], errors="ignore")
        num_samples, num_features = features_df.shape

        categorical_cols = features_df.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_cols = features_df.select_dtypes(include=["number"]).columns.tolist()

        cat_count = len(categorical_cols)
        num_count = len(numerical_cols)
        cat_ratio = cat_count / max(num_features, 1)

        max_card = max([features_df[col].nunique() for col in categorical_cols], default=0)
        missing_ratio = float(features_df.isna().sum().sum()) / max(df.size, 1)

        # Imbalance ratio
        if target_col in df.columns:
            counts = df[target_col].value_counts()
            imbalance = float(counts.min() / max(counts.max(), 1))
        else:
            imbalance = 0.5

        return DatasetMetaFeatures(
            num_samples=num_samples,
            num_features=num_features,
            categorical_count=cat_count,
            numerical_count=num_count,
            categorical_ratio=cat_ratio,
            class_imbalance_ratio=imbalance,
            max_cardinality=max_card,
            missing_cell_ratio=missing_ratio,
        )

    def select_and_lock_classifier(self, meta: DatasetMetaFeatures) -> ClassifierLockDecision:
        """Evaluates meta-features against classifier inductive biases and locks the winner."""
        logger.info("Evaluating dataset meta-features: cat_ratio=%.2f, N=%d", meta.categorical_ratio, meta.num_samples)

        # High categorical ratio on small/medium tabular data decisively favors CatBoost
        if meta.categorical_ratio >= 0.40 and meta.num_samples <= 50000:
            return ClassifierLockDecision(
                selected_family="catboost",
                rationale=(
                    f"Dataset is {meta.categorical_ratio*100:.1f}% categorical with N={meta.num_samples}. "
                    "CatBoost's Ordered Target Statistics prevent leakage on categorical permutations, "
                    "and symmetric oblivious trees provide strong regularization against small tabular overfitting."
                ),
                inductive_bias_fit_score=0.98,
                is_locked=True,
            )
        elif meta.num_samples > 100000 and meta.categorical_ratio < 0.30:
            return ClassifierLockDecision(
                selected_family="lightgbm",
                rationale="Large sample size and dense numerical features favor LightGBM histogram binning.",
                inductive_bias_fit_score=0.92,
                is_locked=True,
            )
        else:
            return ClassifierLockDecision(
                selected_family="catboost",
                rationale="Defaulting to CatBoost as the robust tabular workhorse.",
                inductive_bias_fit_score=0.88,
                is_locked=True,
            )
