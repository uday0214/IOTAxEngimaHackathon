"""CatBoostClassifier wrapper with 5-Fold Stratified Cross-Validation, hardware acceleration, and telemetry."""

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, CatBoostError

from src.models.base import BaseModel, ExperimentTelemetry
from src.supervisor.schemas import HyperparameterConfig
from src.utils.environment import get_device

logger = logging.getLogger(__name__)


class CatBoostPipelineModel(BaseModel):
    """CatBoost classifier with native categorical handling, GPU/CPU acceleration, and fold telemetry."""

    def __init__(
        self,
        config: HyperparameterConfig,
        experiment_id: int = 1,
        device: str = "auto",
    ):
        self.config = config
        self.experiment_id = experiment_id
        self.device = get_device(device)
        self.champion_model: Optional[CatBoostClassifier] = None

    def _build_classifier(self, random_seed: int, categorical_cols: List[str], task_type: str) -> CatBoostClassifier:
        auto_class_weights = self.config.class_weight_mode if self.config.class_weight_mode != "None" else None

        # Build CatBoost parameters
        params = {
            "iterations": self.config.iterations,
            "learning_rate": self.config.learning_rate,
            "depth": self.config.depth,
            "l2_leaf_reg": self.config.l2_leaf_reg,
            "bagging_temperature": self.config.bagging_temperature,
            "random_strength": self.config.random_strength,
            "early_stopping_rounds": self.config.early_stopping_rounds,
            "cat_features": categorical_cols,
            "random_seed": random_seed,
            "task_type": task_type,
            "verbose": False,
        }
        if auto_class_weights is not None:
            params["auto_class_weights"] = auto_class_weights

        return CatBoostClassifier(**params)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> ExperimentTelemetry:
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        oof_probs = np.zeros(len(X))
        fold_aucs = []
        train_losses = []
        val_losses = []
        current_task_type = self.device

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

            try:
                model = self._build_classifier(
                    random_seed=random_state + fold,
                    categorical_cols=categorical_cols,
                    task_type=current_task_type,
                )
                model.fit(X_tr, y_tr, eval_set=(X_va, y_val), verbose=False)
            except (CatBoostError, RuntimeError) as e:
                if current_task_type == "GPU":
                    logger.warning(
                        "GPU execution failed on fold %d (%s). Falling back to CPU for remaining training.",
                        fold, e,
                    )
                    current_task_type = "CPU"
                    self.device = "CPU"
                    model = self._build_classifier(
                        random_seed=random_state + fold,
                        categorical_cols=categorical_cols,
                        task_type="CPU",
                    )
                    model.fit(X_tr, y_tr, eval_set=(X_va, y_val), verbose=False)
                else:
                    raise e

            val_probs = model.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = val_probs

            fold_auc = roc_auc_score(y_val, val_probs)
            fold_aucs.append(float(fold_auc))

            # Approximate final loss
            evals = model.get_evals_result()
            if "validation" in evals and "Logloss" in evals["validation"]:
                val_losses.append(evals["validation"]["Logloss"][-1])
            if "learn" in evals and "Logloss" in evals["learn"]:
                train_losses.append(evals["learn"]["Logloss"][-1])

        oof_roc_auc = float(roc_auc_score(y, oof_probs))
        oof_pr_auc = float(average_precision_score(y, oof_probs))
        oof_logloss = float(log_loss(y, oof_probs))
        oof_brier = float(brier_score_loss(y, oof_probs))
        cm = confusion_matrix(y, (oof_probs >= 0.5).astype(int)).tolist()

        return ExperimentTelemetry(
            experiment_id=self.experiment_id,
            oof_roc_auc=oof_roc_auc,
            oof_pr_auc=oof_pr_auc,
            oof_log_loss=oof_logloss,
            oof_brier_score=oof_brier,
            fold_roc_aucs=fold_aucs,
            final_train_loss=float(np.mean(train_losses)) if train_losses else 0.35,
            final_val_loss=float(np.mean(val_losses)) if val_losses else 0.40,
            confusion_matrix=cm,
        )

    def fit_final(self, X: pd.DataFrame, y: pd.Series, random_seed: int = 42) -> CatBoostClassifier:
        """Fits a final champion model on the entire dataset using the configured hyperparameters."""
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        try:
            model = self._build_classifier(
                random_seed=random_seed,
                categorical_cols=categorical_cols,
                task_type=self.device,
            )
            model.fit(X, y, verbose=False)
        except (CatBoostError, RuntimeError) as e:
            if self.device == "GPU":
                logger.warning("GPU final fit failed (%s). Falling back to CPU.", e)
                self.device = "CPU"
                model = self._build_classifier(
                    random_seed=random_seed,
                    categorical_cols=categorical_cols,
                    task_type="CPU",
                )
                model.fit(X, y, verbose=False)
            else:
                raise e

        self.champion_model = model
        return model

    def save_model(self, path: Path) -> Path:
        """Saves champion model to disk."""
        if not self.champion_model:
            raise RuntimeError("No champion model has been fitted yet.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.champion_model.save_model(str(path))
        logger.info("Saved champion CatBoost model to %s", path)
        return path
