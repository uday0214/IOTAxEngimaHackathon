"""CatBoostClassifier wrapper with 5-Fold Stratified Cross-Validation, hardware acceleration, and telemetry."""

import time
import logging
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, CatBoostError

from src.models.base import BaseModel, ExperimentTelemetry
from src.supervisor.schemas import HyperparameterConfig
from src.telemetry.collector import ExtensiveTelemetryCollector
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
        start_time = time.perf_counter()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        oof_probs = np.zeros(len(X))
        fold_metrics = []
        fold_aucs = []
        train_losses = []
        val_losses = []
        all_evals_train = []
        all_evals_val = []
        best_iterations = []

        feature_names = list(X.columns)
        aggregated_importances = np.zeros(len(feature_names))
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
                        "GPU execution failed on fold %d (%s). Falling back to CPU.",
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

            fold_auc = float(roc_auc_score(y_val, val_probs))
            fold_pr_auc = float(average_precision_score(y_val, val_probs))
            fold_loss = float(log_loss(y_val, val_probs))

            fold_aucs.append(fold_auc)
            fold_metrics.append({
                "fold": fold + 1,
                "roc_auc": round(fold_auc, 5),
                "pr_auc": round(fold_pr_auc, 5),
                "log_loss": round(fold_loss, 4),
            })

            # Feature importance accumulation
            try:
                fi = model.get_feature_importance()
                aggregated_importances += fi
            except Exception:
                pass

            # Best iteration tracking
            best_iter = getattr(model, "get_best_iteration", lambda: None)()
            if best_iter is not None:
                best_iterations.append(int(best_iter))

            # Loss trajectory extraction
            evals = model.get_evals_result()
            if "validation" in evals and "Logloss" in evals["validation"]:
                val_losses.append(evals["validation"]["Logloss"][-1])
                all_evals_val.append(evals["validation"]["Logloss"])
            if "learn" in evals and "Logloss" in evals["learn"]:
                train_losses.append(evals["learn"]["Logloss"][-1])
                all_evals_train.append(evals["learn"]["Logloss"])

        elapsed_time = time.perf_counter() - start_time

        # Calculate rich classification diagnostics via ExtensiveTelemetryCollector
        diag = ExtensiveTelemetryCollector.compute_classification_diagnostics(
            y_true=y.to_numpy(),
            y_prob=oof_probs,
        )

        # Average feature importances across folds
        avg_importances = aggregated_importances / n_splits
        importance_dict = {f: float(imp) for f, imp in zip(feature_names, avg_importances)}
        top_f, bottom_f, zero_f = ExtensiveTelemetryCollector.process_feature_importance_spectrum(importance_dict)

        # Average loss curves
        mean_best_iter = int(np.mean(best_iterations)) if best_iterations else self.config.iterations
        if all_evals_val and all_evals_train:
            min_len = min(len(curve) for curve in all_evals_val)
            mean_train_curve = np.mean([c[:min_len] for c in all_evals_train], axis=0).tolist()
            mean_val_curve = np.mean([c[:min_len] for c in all_evals_val], axis=0).tolist()
            sampled_losses, tail_slope = ExtensiveTelemetryCollector.sample_loss_trajectories(
                mean_train_curve, mean_val_curve, n_samples=15
            )
        else:
            sampled_losses, tail_slope = [], 0.0

        final_tr_loss = float(np.mean(train_losses)) if train_losses else 0.35
        final_va_loss = float(np.mean(val_losses)) if val_losses else 0.40
        gen_gap = final_va_loss - final_tr_loss
        overfit_idx = (gen_gap / max(final_va_loss, 1e-6)) * 100.0

        return ExperimentTelemetry(
            experiment_id=self.experiment_id,
            oof_roc_auc=diag["oof_roc_auc"],
            oof_pr_auc=diag["oof_pr_auc"],
            oof_log_loss=diag["oof_log_loss"],
            oof_brier_score=diag["oof_brier_score"],
            fold_roc_aucs=fold_aucs,
            fold_metrics=fold_metrics,
            fold_std_roc_auc=float(np.std(fold_aucs)),
            final_train_loss=final_tr_loss,
            final_val_loss=final_va_loss,
            generalization_gap=gen_gap,
            overfitting_index=overfit_idx,
            best_iteration=mean_best_iter,
            total_iterations=self.config.iterations,
            early_stopping_fraction=mean_best_iter / max(self.config.iterations, 1),
            val_loss_gradient_tail=tail_slope,
            loss_trajectory_samples=sampled_losses,
            feature_importances=importance_dict,
            top_features=top_f,
            bottom_features=bottom_f,
            zero_importance_features=zero_f,
            ece_score=diag["ece_score"],
            optimal_youden_threshold=diag["optimal_youden_threshold"],
            f1_macro_05=diag["f1_macro_05"],
            f1_at_optimal=diag["f1_at_optimal"],
            matthews_corrcoef=diag["matthews_corrcoef"],
            confusion_matrix_05=diag["confusion_matrix_05"],
            confusion_matrix_optimal=diag["confusion_matrix_optimal"],
            false_negative_rate=diag["false_negative_rate"],
            false_positive_rate=diag["false_positive_rate"],
            training_duration_seconds=round(elapsed_time, 2),
            device_used=current_task_type,
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
