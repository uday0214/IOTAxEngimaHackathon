"""Dynamic feature transformation pipeline controlled by Supervisor configs."""

import pandas as pd
from src.supervisor.schemas import FeatureEngineeringConfig
from src.features.candidate_library import ChurnFeatureLibrary


class DynamicFeaturePipeline:
    """Applies candidate feature transformations based on Supervisor configuration."""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.config.enable_charge_ratios:
            df = ChurnFeatureLibrary.add_charge_ratios(df)

        if self.config.enable_tenure_cohorts:
            df = ChurnFeatureLibrary.add_tenure_cohorts(df)

        if self.config.enable_service_density:
            df = ChurnFeatureLibrary.add_service_density(df)

        if self.config.enable_interaction_terms:
            df = ChurnFeatureLibrary.add_interaction_terms(df)

        # Feature pruning
        if self.config.drop_low_importance_features:
            df = df.drop(columns=self.config.drop_low_importance_features, errors="ignore")

        return df
