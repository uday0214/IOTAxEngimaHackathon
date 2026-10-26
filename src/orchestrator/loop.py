"""Main Autonomous Orchestrator: Coordinates Pre-Flight and 15-Run Training Loop."""

import logging
from pathlib import Path
from typing import Optional
import yaml
import pandas as pd

from src.budget.guard import BudgetGuard, BudgetExhaustedError
from src.data.loader import DataLoader
from src.data.validator import DataProvenanceGuard
from src.supervisor.profiler import PreFlightProfiler
from src.supervisor.math_engine import MathematicalInferenceEngine
from src.supervisor.llm_client import GeminiSupervisorClient
from src.supervisor.prompt_builder import SupervisorPromptBuilder
from src.supervisor.schemas import SupervisorDecision, HyperparameterConfig, FeatureEngineeringConfig
from src.features.pipeline import DynamicFeaturePipeline
from src.models.catboost_model import CatBoostPipelineModel
from src.telemetry.ledger import ExperimentLedger

logger = logging.getLogger(__name__)


class AutonomousExperimentLoop:
    """Executes the autonomous loop up to exactly 15 experiments."""

    def __init__(
        self,
        max_runs: int = 15,
        ledger_path: Path = Path("artifacts/experiment_ledger.jsonl"),
        manifest_path: Path = Path("artifacts/dataset_manifest.json"),
        canonical_path: Path = Path("data/processed/train_val.parquet"),
    ):
        self.max_runs = max_runs
        self.budget_guard = BudgetGuard(ledger_path=ledger_path, max_experiments=max_runs)
        self.data_guard = DataProvenanceGuard(manifest_path=manifest_path, canonical_data_path=canonical_path)
        self.ledger = ExperimentLedger(ledger_path=ledger_path)
        self.profiler = PreFlightProfiler()
        self.math_engine = MathematicalInferenceEngine()
        self.prompt_builder = SupervisorPromptBuilder()
        self.llm_client = GeminiSupervisorClient()
        self.data_loader = DataLoader()

    def run_pre_flight(self, train_val_df: pd.DataFrame) -> str:
        """Step 0: Profiles dataset and locks best classifier with zero experiment cost."""
        logger.info("Executing Pre-Flight Profiling (Cost: 0 runs)...")
        meta = self.profiler.extract_meta_features(train_val_df, target_col="Churn")
        decision = self.profiler.select_and_lock_classifier(meta)
        logger.info("Pre-Flight Decision: Locked classifier '%s' (Fit score: %.2f)", decision.selected_family, decision.inductive_bias_fit_score)
        return decision.selected_family

    def run_experiment(self, experiment_id: int, config: SupervisorDecision, train_val_df: pd.DataFrame) -> dict:
        """Executes a single model fit-and-validation experiment under budget guard."""
        # 1. Budget Guard Check
        self.budget_guard.register_fit_start(experiment_id)

        # 2. Data Provenance Audit
        verified_df = self.data_guard.assert_strictly_canonical(train_val_df)

        # 3. Dynamic Feature Transformation
        pipeline = DynamicFeaturePipeline(config.feature_config)
        transformed_df = pipeline.transform(verified_df)

        X = transformed_df.drop(columns=["Churn"], errors="ignore")
        y = transformed_df["Churn"]

        # 4. Fit & 5-Fold Stratified Cross-Validation (Counts as 1 Experiment)
        model = CatBoostPipelineModel(config=config.hyperparameters, experiment_id=experiment_id)
        telemetry = model.cross_validate(X, y, n_splits=5)

        # 5. Persist run to ledger
        entry = {
            "experiment_id": experiment_id,
            "hypothesis": config.hypothesis,
            "hyperparameters": config.hyperparameters.model_dump(),
            "feature_config": config.feature_config.model_dump(),
            "metrics": {
                "oof_roc_auc": telemetry.oof_roc_auc,
                "oof_pr_auc": telemetry.oof_pr_auc,
                "oof_log_loss": telemetry.oof_log_loss,
                "final_train_loss": telemetry.final_train_loss,
                "final_val_loss": telemetry.final_val_loss,
            },
        }
        self.ledger.append_run(entry)
        logger.info(
            "Completed Run %d/15: OOF ROC-AUC=%.5f, PR-AUC=%.5f",
            experiment_id, telemetry.oof_roc_auc, telemetry.oof_pr_auc
        )
        return entry
