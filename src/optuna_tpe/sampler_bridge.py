"""Bridges Supervisor curated parameter regions into single-trial Optuna steps."""

import logging
from typing import Dict, Any, Optional
import optuna
from src.supervisor.schemas import HyperparameterConfig

logger = logging.getLogger(__name__)


class OptunaSamplerBridge:
    """Manages an Optuna Study constrained to exactly 1 trial per experiment."""

    def __init__(self, study_name: str = "aads_study", storage: str = "sqlite:///artifacts/optuna_study.db"):
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
        )
        self.current_trial: Optional[optuna.trial.Trial] = None

    def sample_parameters(self, supervisor_params: HyperparameterConfig, trial_id: int) -> Dict[str, Any]:
        """Samples refined parameters around supervisor-indicated centers using TPE."""
        self.current_trial = self.study.ask()
        trial = self.current_trial
        family = supervisor_params.model_family

        if family == "logistic_regression":
            c_val = getattr(supervisor_params, "C", 1.0) or 1.0
            return {
                "C": trial.suggest_float("lr_C", max(0.01, c_val * 0.5), min(50.0, c_val * 2.0), log=True),
                "penalty": getattr(supervisor_params, "penalty", "l2") or "l2",
                "solver": getattr(supervisor_params, "solver", "lbfgs") or "lbfgs",
                "class_weight_mode": supervisor_params.class_weight_mode,
            }
        elif family == "random_forest":
            c_depth = supervisor_params.depth
            min_split = getattr(supervisor_params, "min_samples_split", 2) or 2
            return {
                "depth": trial.suggest_int("rf_depth", max(3, c_depth - 1), min(12, c_depth + 2)),
                "min_samples_split": trial.suggest_int("rf_min_split", max(2, min_split - 1), min(10, min_split + 2)),
                "iterations": min(supervisor_params.iterations, 500),
                "class_weight_mode": supervisor_params.class_weight_mode,
            }
        else:
            # Tree boosting (catboost, lightgbm, xgboost)
            c_lr = supervisor_params.learning_rate
            c_depth = supervisor_params.depth
            c_l2 = supervisor_params.l2_leaf_reg

            sampled: Dict[str, Any] = {
                "learning_rate": trial.suggest_float(f"{family}_lr", max(0.005, c_lr * 0.8), min(0.2, c_lr * 1.2), log=True),
                "depth": trial.suggest_int(f"{family}_depth", max(3, c_depth - 1), min(8, c_depth + 1)),
                "l2_leaf_reg": trial.suggest_float(f"{family}_l2", max(0.5, c_l2 * 0.7), min(30.0, c_l2 * 1.3)),
                "iterations": supervisor_params.iterations,
                "early_stopping_rounds": supervisor_params.early_stopping_rounds,
                "class_weight_mode": supervisor_params.class_weight_mode,
            }
            if family == "catboost":
                sampled["bagging_temperature"] = trial.suggest_float("cb_bagging_temp", 0.0, 1.0)
            else:
                sampled["subsample"] = trial.suggest_float(f"{family}_subsample", 0.6, 1.0)
                sampled["colsample_bytree"] = trial.suggest_float(f"{family}_colsample", 0.6, 1.0)
            return sampled

    def record_result(self, trial_number: int, roc_auc: float) -> None:
        """Tell the study the result of the completed experiment trial."""
        try:
            if self.current_trial is not None:
                self.study.tell(self.current_trial, roc_auc)
                self.current_trial = None
            else:
                self.study.tell(trial_number, roc_auc)
        except Exception as e:
            logger.warning("Could not record trial %d in Optuna study: %s", trial_number, e)

    def close(self) -> None:
        """Closes the underlying RDB storage engine if applicable."""
        storage = getattr(self.study, "_storage", None)
        if storage is not None and hasattr(storage, "engine"):
            try:
                storage.engine.dispose()
            except Exception:
                pass
