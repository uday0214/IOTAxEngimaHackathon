"""Pydantic data schemas for Gemini Supervisor Agent and Experiment Configurations."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class HyperparameterConfig(BaseModel):
    """Hyperparameter specification for the classifier."""
    model_family: Literal["catboost", "lightgbm", "xgboost"] = Field(
        default="catboost", description="Selected classifier family"
    )
    learning_rate: float = Field(default=0.05, ge=0.005, le=0.3)
    depth: int = Field(default=6, ge=3, le=10)
    l2_leaf_reg: float = Field(default=3.0, ge=0.1, le=50.0)
    iterations: int = Field(default=1000, ge=300, le=3000)
    subsample: Optional[float] = Field(default=0.8, ge=0.4, le=1.0)
    bagging_temperature: Optional[float] = Field(default=0.2, ge=0.0, le=1.0)
    random_strength: Optional[float] = Field(default=1.0, ge=0.0, le=10.0)
    class_weight_mode: Literal["None", "Balanced", "SqrtBalanced"] = "None"
    early_stopping_rounds: int = Field(default=50, ge=10, le=100)
    random_seed: int = 42


class FeatureEngineeringConfig(BaseModel):
    """Feature generation and pruning toggles."""
    enable_charge_ratios: bool = Field(default=True, description="Monthly-to-total charge velocity")
    enable_tenure_cohorts: bool = Field(default=True, description="Tenure hazard groups")
    enable_service_density: bool = Field(default=True, description="Total services count & streaming ratio")
    enable_interaction_terms: bool = Field(default=False, description="Contract x payment interactions")
    drop_low_importance_features: List[str] = Field(default_factory=list, description="Features to prune")


class SupervisorDecision(BaseModel):
    """Structured decision emitted by Gemini/OpenRouter Supervisor for the next experiment."""
    experiment_id: int = Field(..., ge=1, le=15, description="Next experiment number (1-15)")
    hypothesis: str = Field(..., description="Scientific hypothesis explaining the proposed changes")
    mathematical_justification: str = Field(
        ..., description="Explicit references to Overfitting Index, DeLong p-value, or SHAP stats"
    )
    hyperparameters: HyperparameterConfig
    feature_config: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig, description="Feature engineering and pruning toggles"
    )
    expected_outcome: str = Field(
        default="Targeting ROC-AUC and PR-AUC improvement with regularized generalization.",
        description="Anticipated directional impact on ROC-AUC and PR-AUC",
    )
