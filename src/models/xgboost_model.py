"""XGBoost classifier wrapper with 5-Fold Stratified Cross-Validation, telemetry, and persistence."""

import time
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.models.base import BaseModel, ExperimentTelemetry
from src.supervisor.schemas import HyperparameterConfig
from src.telemetry.collector import ExtensiveTelemetryCollector
from src.utils.environment import get_device

logger = logging.getLogger(__name__)


class XGBoostPipelineModel(BaseModel):
    """XGBoost classifier with native categorical support, 5-fold cross-validation, and telemetry."""

    def __init__(
        self,
        config: HyperparameterConfig,
        experiment_id: int = 1,
        device: str = "auto",
        cat_features: Optional[List[str]] = None,
    ):
        self.config = config
        self.experiment_id = experiment_id
        self.device = get_device(device)
        self.cat_features = cat_features
        self.champion_model: Optional[XGBClassifier] = None

    def _prepare_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensures categorical columns are cast to category dtype with missing values handled."""
        X_df = X.copy()
        cat_cols = self.cat_features if self.cat_features is not None else [
            col for col in X_df.columns
            if X_df[col].dtype.name in ("object", "category", "string")
            or isinstance(X_df[col].dtype, (pd.CategoricalDtype, pd.StringDtype))
        ]
        for col in cat_cols:
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna("Missing").astype("category")
        return X_df

    def _build_classifier(self, random_seed: int, with_early_stopping: bool = True) -> XGBClassifier:
        n_est = self.config.n_estimators or self.config.iterations
        colsample = getattr(self.config, "colsample_bytree", 0.8) or 0.8
        scale_pos = 2.8 if self.config.class_weight_mode in ("Balanced", "SqrtBalanced") else 1.0

        params: Dict[str, Any] = {
            "n_estimators": n_est,
            "learning_rate": self.config.learning_rate,
            "max_depth": self.config.depth,
            "reg_lambda": self.config.l2_leaf_reg,
            "subsample": self.config.subsample or 0.8,
            "colsample_bytree": colsample,
            "scale_pos_weight": scale_pos,
            "eval_metric": "logloss",
            "enable_categorical": True,
            "tree_method": "hist",
            "random_state": random_seed,
            "n_jobs": -1,
        }
        if with_early_stopping:
            params["early_stopping_rounds"] = self.config.early_stopping_rounds

        if self.device == "GPU":
            params["device"] = "cuda"
        else:
            params["device"] = "cpu"

        return XGBClassifier(**params)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> ExperimentTelemetry:
        start_time = time.perf_counter()
        X_proc = self._prepare_dataframe(X)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        oof_probs = np.zeros(len(X_proc))
        fold_metrics = []
        fold_aucs = []
        train_losses = []
        val_losses = []
        best_iterations = []

        feature_names = list(X_proc.columns)
        aggregated_importances = np.zeros(len(feature_names))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_proc, y)):
            X_tr, y_tr = X_proc.iloc[train_idx], y.iloc[train_idx]
            X_va, y_val = X_proc.iloc[val_idx], y.iloc[val_idx]

            model = self._build_classifier(random_seed=random_state + fold, with_early_stopping=True)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    model.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_tr, y_tr), (X_va, y_val)],
                        verbose=False,
                    )
            except Exception as e:
                if self.device == "GPU":
                    logger.warning("XGBoost GPU fit failed on fold %d (%s). Falling back to CPU.", fold, e)
                    self.device = "CPU"
                    model = self._build_classifier(random_seed=random_state + fold, with_early_stopping=True)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        model.fit(
                            X_tr,
                            y_tr,
                            eval_set=[(X_tr, y_tr), (X_va, y_val)],
                            verbose=False,
                        )
                else:
                    raise e

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
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

            # Feature importance
            try:
                fi = model.feature_importances_
                aggregated_importances += fi
            except Exception:
                pass

            # Best iteration tracking
            best_iter = getattr(model, "best_iteration", None)
            if best_iter is not None:
                best_iterations.append(int(best_iter))

            # Train / Val loss extraction
            evals_res = getattr(model, "evals_result_", {})
            if "validation_1" in evals_res and "logloss" in evals_res["validation_1"]:
                val_losses.append(evals_res["validation_1"]["logloss"][-1])
            if "validation_0" in evals_res and "logloss" in evals_res["validation_0"]:
                train_losses.append(evals_res["validation_0"]["logloss"][-1])

        elapsed_time = time.perf_counter() - start_time

        # Calculate comprehensive diagnostics
        diag = ExtensiveTelemetryCollector.compute_classification_diagnostics(
            y_true=y.to_numpy(),
            y_prob=oof_probs,
            X_df=X,
        )

        # Average feature importances
        avg_importances = aggregated_importances / n_splits
        imp_sum = sum(avg_importances) if sum(avg_importances) > 0 else 1.0
        normalized_imp = (avg_importances / imp_sum) * 100.0
        importance_dict = {f: float(imp) for f, imp in zip(feature_names, normalized_imp)}
        top_f, bottom_f, zero_f = ExtensiveTelemetryCollector.process_feature_importance_spectrum(importance_dict)

        total_iters = self.config.n_estimators or self.config.iterations
        mean_best_iter = int(np.mean(best_iterations)) if best_iterations else total_iters
        final_tr_loss = float(np.mean(train_losses)) if train_losses else 0.35
        final_va_loss = float(np.mean(val_losses)) if val_losses else 0.40
        gen_gap = final_va_loss - final_tr_loss
        overfit_idx = (gen_gap / max(final_va_loss, 1e-6)) * 100.0

        throughput = len(X) / max(elapsed_time, 1e-4)
        inf_lat_ms = (elapsed_time / max(len(X) * n_splits, 1)) * 1000.0
        mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 1.0
        fold_cv = (float(np.std(fold_aucs)) / max(mean_auc, 1e-6)) * 100.0

        return ExperimentTelemetry(
            experiment_id=self.experiment_id,
            oof_roc_auc=diag["oof_roc_auc"],
            oof_pr_auc=diag["oof_pr_auc"],
            oof_log_loss=diag["oof_log_loss"],
            oof_brier_score=diag["oof_brier_score"],
            fold_roc_aucs=fold_aucs,
            fold_metrics=fold_metrics,
            fold_std_roc_auc=float(np.std(fold_aucs)),
            fold_cv_roc_auc=round(fold_cv, 3),
            final_train_loss=final_tr_loss,
            final_val_loss=final_va_loss,
            generalization_gap=gen_gap,
            overfitting_index=overfit_idx,
            best_iteration=mean_best_iter,
            total_iterations=total_iters,
            early_stopping_fraction=mean_best_iter / max(total_iters, 1),
            val_loss_gradient_tail=0.0,
            loss_trajectory_samples=[],
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
            roc_curve_points=diag.get("roc_curve_points", []),
            pr_curve_points=diag.get("pr_curve_points", []),
            calibration_curve_points=diag.get("calibration_curve_points", []),
            probability_histogram=diag.get("probability_histogram", []),
            training_duration_seconds=round(elapsed_time, 2),
            device_used=self.device,
            throughput_samples_per_sec=round(throughput, 2),
            inference_latency_ms=round(inf_lat_ms, 3),
            threshold_sweep=diag.get("threshold_sweep", []),
            lift_and_gain_curve=diag.get("lift_and_gain_curve", []),
            distribution_stats=diag.get("distribution_stats", {}),
            separation_margin=diag.get("separation_margin", 0.0),
            cohort_slice_metrics=diag.get("cohort_slice_metrics", {}),
        )

    def fit_final(self, X: pd.DataFrame, y: pd.Series, random_seed: int = 42) -> XGBClassifier:
        """Fits final champion model on full dataset."""
        X_proc = self._prepare_dataframe(X)
        model = self._build_classifier(random_seed=random_seed, with_early_stopping=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model.fit(X_proc, y, verbose=False)
        self.champion_model = model
        return model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts churn probabilities using the fitted model."""
        if not self.champion_model:
            raise RuntimeError("Champion model is not fitted or loaded yet.")
        X_proc = self._prepare_dataframe(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return self.champion_model.predict_proba(X_proc)[:, 1]

    def save_model(self, path: Path) -> Path:
        """Saves champion model to disk via joblib."""
        if not self.champion_model:
            raise RuntimeError("No champion model has been fitted yet.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.champion_model, path)
        logger.info("Saved champion XGBoost model to %s", path)
        return path

    @classmethod
    def load_model(cls, path: Path, config: Optional[HyperparameterConfig] = None) -> "XGBoostPipelineModel":
        """Loads a persisted XGBoost model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        instance = cls(config=config or HyperparameterConfig(model_family="xgboost"))
        instance.champion_model = joblib.load(path)
        return instance
