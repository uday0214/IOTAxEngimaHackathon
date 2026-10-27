"""Base classifier interface and comprehensive telemetry data models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd


@dataclass
class ExperimentTelemetry:
    """Extensive telemetry collected during and after each fit-and-validation experiment."""

    experiment_id: int
    oof_roc_auc: float
    oof_pr_auc: float
    oof_log_loss: float
    oof_brier_score: float

    # Fold-level granularity
    fold_roc_aucs: List[float] = field(default_factory=list)
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    fold_std_roc_auc: float = 0.0

    # Learning curve dynamics & iterations
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    generalization_gap: float = 0.0
    overfitting_index: float = 0.0
    best_iteration: int = 0
    total_iterations: int = 1000
    early_stopping_fraction: float = 1.0
    val_loss_gradient_tail: float = 0.0
    loss_trajectory_samples: List[Dict[str, Any]] = field(default_factory=list)

    # Feature importance spectrum
    feature_importances: Dict[str, float] = field(default_factory=dict)
    top_features: List[Dict[str, Any]] = field(default_factory=list)
    bottom_features: List[Dict[str, Any]] = field(default_factory=list)
    zero_importance_features: List[str] = field(default_factory=list)

    # Probability calibration & decision thresholding
    ece_score: float = 0.0
    optimal_youden_threshold: float = 0.5
    f1_macro_05: float = 0.0
    f1_at_optimal: float = 0.0
    matthews_corrcoef: float = 0.0

    # Curve points for visualization (ROC, PR, Calibration, Histogram)
    roc_curve_points: List[Dict[str, float]] = field(default_factory=list)
    pr_curve_points: List[Dict[str, float]] = field(default_factory=list)
    calibration_curve_points: List[Dict[str, float]] = field(default_factory=list)
    probability_histogram: List[Dict[str, Any]] = field(default_factory=list)

    # Error analysis & confusion matrices
    confusion_matrix_05: List[List[int]] = field(default_factory=list)
    confusion_matrix_optimal: List[List[int]] = field(default_factory=list)
    false_negative_rate: float = 0.0
    false_positive_rate: float = 0.0

    # Hardware & performance profiling
    training_duration_seconds: float = 0.0
    device_used: str = "CPU"
    throughput_samples_per_sec: float = 0.0
    inference_latency_ms: float = 0.0
    fold_cv_roc_auc: float = 0.0

    # Multi-threshold decision & business spectrum
    threshold_sweep: List[Dict[str, Any]] = field(default_factory=list)

    # Decile cumulative gains & lift analysis
    lift_and_gain_curve: List[Dict[str, Any]] = field(default_factory=list)

    # Prediction probability distribution moments & separation
    distribution_stats: Dict[str, Any] = field(default_factory=dict)
    separation_margin: float = 0.0

    # Subgroup & cohort slice diagnostics
    cohort_slice_metrics: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract interface for all classifier models."""

    @abstractmethod
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> ExperimentTelemetry:
        """Executes n-fold stratified cross-validation and records extensive telemetry."""
        pass

    @abstractmethod
    def fit_final(self, X: pd.DataFrame, y: pd.Series, random_seed: int = 42) -> Any:
        """Fits a final champion model on the entire dataset."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generates class probability predictions."""
        pass

    @abstractmethod
    def save_model(self, path: Any) -> Any:
        """Saves champion model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path: Any, config: Optional[Any] = None) -> "BaseModel":
        """Loads a persisted model from disk."""
        pass
