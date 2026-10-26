"""Base classifier interface and telemetry return types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class ExperimentTelemetry:
    experiment_id: int
    oof_roc_auc: float
    oof_pr_auc: float
    oof_log_loss: float
    oof_brier_score: float
    fold_roc_aucs: List[float] = field(default_factory=list)
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    iteration_train_losses: List[float] = field(default_factory=list)
    iteration_val_losses: List[float] = field(default_factory=list)
    feature_importances: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    youden_threshold: float = 0.5


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
        """Executes n-fold stratified cross-validation and records telemetry."""
        pass
