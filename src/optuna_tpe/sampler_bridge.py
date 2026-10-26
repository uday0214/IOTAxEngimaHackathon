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
        # Create an Optuna trial manually
        self.current_trial = self.study.ask()
        trial = self.current_trial

        # Narrow sampling around Supervisor targets
        c_lr = supervisor_params.learning_rate
        c_depth = supervisor_params.depth
        c_l2 = supervisor_params.l2_leaf_reg

        sampled = {
            "learning_rate": trial.suggest_float("learning_rate", max(0.005, c_lr * 0.8), min(0.2, c_lr * 1.2), log=True),
            "depth": trial.suggest_int("depth", max(3, c_depth - 1), min(8, c_depth + 1)),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", max(0.5, c_l2 * 0.7), min(30.0, c_l2 * 1.3)),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "iterations": supervisor_params.iterations,
            "early_stopping_rounds": supervisor_params.early_stopping_rounds,
            "class_weight_mode": supervisor_params.class_weight_mode,
        }
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
