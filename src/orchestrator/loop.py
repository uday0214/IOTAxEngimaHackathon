"""Main Autonomous Orchestrator: End-to-end agentic loop for model training."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from src.budget.guard import BudgetGuard, BudgetExhaustedError
from src.data.loader import DataLoader
from src.data.validator import DataProvenanceGuard
from src.supervisor.profiler import PreFlightProfiler, ClassifierLockDecision
from src.supervisor.math_engine import MathematicalInferenceEngine, MathDiagnosis
from src.supervisor.llm_client import GeminiSupervisorClient
from src.supervisor.prompt_builder import SupervisorPromptBuilder
from src.supervisor.schemas import SupervisorDecision, HyperparameterConfig, FeatureEngineeringConfig
from src.features.pipeline import DynamicFeaturePipeline
from src.optuna_tpe.sampler_bridge import OptunaSamplerBridge
from src.telemetry.ledger import ExperimentLedger
from src.models.base import ExperimentTelemetry
from src.utils.environment import get_device

logger = logging.getLogger(__name__)


class AutonomousExperimentLoop:
    """Orchestrates the 15-iteration supervisor-directed training and optimization loop."""

    def __init__(
        self,
        max_runs: int = 15,
        ledger_path: Path = Path("artifacts/experiment_ledger.jsonl"),
        manifest_path: Path = Path("artifacts/dataset_manifest.json"),
        canonical_path: Path = Path("data/processed/train_val.parquet"),
        execute_live_training: bool = False,
        device: str = "auto",
    ):
        """
        Args:
            max_runs: Strict budget limit (15).
            ledger_path: Path to append-only JSONL ledger.
            manifest_path: Path to dataset cryptographic manifest.
            canonical_path: Path to verified train_val dataset partition.
            execute_live_training: If True, invokes actual CatBoost training;
                                  If False, runs simulated telemetry (enables testing
                                  the full loop without loading model weights yet).
            device: Target hardware device ('auto', 'cpu', 'gpu').
        """
        self.max_runs = max_runs
        self.execute_live_training = execute_live_training
        self.device = get_device(device)

        # Core System Components
        self.budget_guard = BudgetGuard(ledger_path=ledger_path, max_experiments=max_runs)
        self.data_guard = DataProvenanceGuard(manifest_path=manifest_path, canonical_data_path=canonical_path)
        self.ledger = ExperimentLedger(ledger_path=ledger_path)
        self.profiler = PreFlightProfiler()
        self.math_engine = MathematicalInferenceEngine()
        self.prompt_builder = SupervisorPromptBuilder()
        self.llm_client = GeminiSupervisorClient()
        self.optuna_bridge = OptunaSamplerBridge()
        self.data_loader = DataLoader()

        self.locked_classifier: Optional[ClassifierLockDecision] = None

    def run_pre_flight(self, train_val_df: pd.DataFrame) -> ClassifierLockDecision:
        """Step 0: Profiles dataset and locks best classifier with ZERO experiment cost."""
        logger.info("=== PRE-FLIGHT PHASE: Profiling Dataset (Budget Cost: 0) ===")
        meta = self.profiler.extract_meta_features(train_val_df, target_col="Churn")
        self.locked_classifier = self.profiler.select_and_lock_classifier(meta)
        logger.info(
            "Classifier Selection Locked: '%s' (Fit Score: %.2f) | Rationale: %s",
            self.locked_classifier.selected_family,
            self.locked_classifier.inductive_bias_fit_score,
            self.locked_classifier.rationale,
        )
        return self.locked_classifier

    def run_single_experiment(
        self,
        experiment_id: int,
        decision: SupervisorDecision,
        train_val_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Executes a single model fit-and-validation experiment under the budget guard."""
        # 1. Physical Budget Guard Verification
        self.budget_guard.register_fit_start(experiment_id)

        # 2. Strict Dataset Provenance Audit
        verified_df = self.data_guard.assert_strictly_canonical(train_val_df)

        # 3. Dynamic Feature Transformation Pipeline
        feature_pipeline = DynamicFeaturePipeline(decision.feature_config)
        transformed_df = feature_pipeline.transform(verified_df)

        X = transformed_df.drop(columns=["Churn"], errors="ignore")
        y = transformed_df["Churn"] if "Churn" in transformed_df.columns else pd.Series(np.zeros(len(X)))

        # 4. Optuna TPE Guided Sampling (Maps Supervisor bounds into single trial step)
        sampled_params = self.optuna_bridge.sample_parameters(
            supervisor_params=decision.hyperparameters,
            trial_id=experiment_id,
        )

        # 5. Fit & Validation Execution
        if self.execute_live_training:
            from src.models.catboost_model import CatBoostPipelineModel
            logger.info("Live Training: Fitting CatBoost 5-fold CV for Experiment %d (device: %s)...", experiment_id, self.device)
            model = CatBoostPipelineModel(
                config=decision.hyperparameters,
                experiment_id=experiment_id,
                device=self.device,
            )
            telemetry = model.cross_validate(X, y, n_splits=5)
        else:
            logger.info("Loop Simulation Mode: Simulating telemetry for Experiment %d (Model not loaded yet)...", experiment_id)
            telemetry = self._simulate_telemetry(experiment_id, decision)

        # 6. Inform Optuna study of result
        self.optuna_bridge.record_result(experiment_id - 1, telemetry.oof_roc_auc)

        # 7. Record Experiment Ledger Entry
        entry = {
            "experiment_id": experiment_id,
            "hypothesis": decision.hypothesis,
            "mathematical_justification": decision.mathematical_justification,
            "hyperparameters": decision.hyperparameters.model_dump(),
            "sampled_params": sampled_params,
            "feature_config": decision.feature_config.model_dump(),
            "metrics": {
                "oof_roc_auc": telemetry.oof_roc_auc,
                "oof_pr_auc": telemetry.oof_pr_auc,
                "oof_log_loss": telemetry.oof_log_loss,
                "oof_brier_score": telemetry.oof_brier_score,
                "final_train_loss": telemetry.final_train_loss,
                "final_val_loss": telemetry.final_val_loss,
                "fold_roc_aucs": telemetry.fold_roc_aucs,
            },
            "expected_outcome": decision.expected_outcome,
        }
        self.ledger.append_run(entry)
        logger.info(
            "--> Run %d/15 Recorded: ROC-AUC=%.5f | PR-AUC=%.5f | LogLoss=%.4f",
            experiment_id, telemetry.oof_roc_auc, telemetry.oof_pr_auc, telemetry.oof_log_loss
        )
        return entry

    def execute_full_15_run_loop(self, train_val_df: pd.DataFrame) -> Dict[str, Any]:
        """Runs the entire 15-iteration supervisor feedback loop from start to finish."""
        logger.info("=================================================================")
        logger.info("LAUNCHING AUTONOMOUS DATA SCIENTIST LOOP (15 Experiment Budget)")
        logger.info("=================================================================")

        # Step 0: Pre-Flight Profiling (Zero Cost)
        self.run_pre_flight(train_val_df)

        # Baseline Initial Configuration for Experiment 1
        current_decision = SupervisorDecision(
            experiment_id=1,
            hypothesis="Baseline Run 1: Establish benchmark performance with default CatBoost hyperparameters and raw features.",
            mathematical_justification="Anchor point to measure baseline generalization gap and initial loss trajectory.",
            hyperparameters=HyperparameterConfig(
                model_family="catboost",
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                iterations=1000,
            ),
            feature_config=FeatureEngineeringConfig(
                enable_charge_ratios=False,
                enable_tenure_cohorts=False,
                enable_service_density=False,
                enable_interaction_terms=False,
            ),
            expected_outcome="Establish baseline OOF ROC-AUC (~0.84) and initial validation loss.",
        )

        completed_entries = []

        for exp_id in range(1, self.max_runs + 1):
            if self.budget_guard.is_exhausted():
                logger.warning("Budget limit reached (%d runs). Terminating loop.", self.max_runs)
                break

            logger.info("-----------------------------------------------------------------")
            logger.info(">>> STARTING EXPERIMENT %d OF %d <<<", exp_id, self.max_runs)
            logger.info("Hypothesis: %s", current_decision.hypothesis)

            # Execute the single experiment
            run_result = self.run_single_experiment(exp_id, current_decision, train_val_df)
            completed_entries.append(run_result)

            # If more experiments remain, compute mathematical feedback and invoke Gemini Supervisor
            if exp_id < self.max_runs:
                best_run = self.ledger.get_best_run()
                best_auc = best_run["metrics"]["oof_roc_auc"] if best_run else run_result["metrics"]["oof_roc_auc"]

                # 1. Mathematical Diagnostics
                diagnosis = self.math_engine.analyze(
                    current_telemetry=run_result["metrics"],
                    best_telemetry=best_run["metrics"] if best_run else None,
                    ledger_history=completed_entries,
                )
                logger.info(
                    "Math Diagnosis for Exp %d: Overfitting Index=%.2f%%, Rec=%s",
                    exp_id, diagnosis.overfitting_index, diagnosis.recommended_action
                )

                # 2. Build Structured Context Prompt
                prompt = self.prompt_builder.build_prompt(
                    experiment_id=exp_id + 1,
                    history=self.ledger.load_all_runs(),
                    current_telemetry=run_result["metrics"],
                    math_diagnosis=diagnosis,
                    best_auc=best_auc,
                )

                # 3. Gemini Supervisor Query
                logger.info("Consulting Gemini Supervisor for Experiment %d config...", exp_id + 1)
                current_decision = self.llm_client.decide_next_experiment(prompt, experiment_id=exp_id + 1)

        best_overall = self.ledger.get_best_run()
        logger.info("=================================================================")
        logger.info("AUTONOMOUS LOOP COMPLETED: 15/15 EXPERIMENTS RECORDED")
        if best_overall:
            logger.info(
                "Best Experiment: Run #%d | Best OOF ROC-AUC: %.5f | PR-AUC: %.5f",
                best_overall["experiment_id"],
                best_overall["metrics"]["oof_roc_auc"],
                best_overall["metrics"]["oof_pr_auc"],
            )

        # Persist champion model artifact if live training was executed
        champion_path = Path("artifacts/champion_model.cbm")
        if self.execute_live_training and best_overall:
            try:
                from src.models.catboost_model import CatBoostPipelineModel
                best_config = HyperparameterConfig(**best_overall["hyperparameters"])
                best_feature_cfg = FeatureEngineeringConfig(**best_overall["feature_config"])
                pipeline = DynamicFeaturePipeline(best_feature_cfg)
                transformed_df = pipeline.transform(train_val_df)
                X_all = transformed_df.drop(columns=["Churn"], errors="ignore")
                y_all = transformed_df["Churn"] if "Churn" in transformed_df.columns else pd.Series(np.zeros(len(X_all)))
                champion = CatBoostPipelineModel(config=best_config, device=self.device)
                champion.fit_final(X_all, y_all)
                champion.save_model(champion_path)
                logger.info("Persisted final champion model to %s", champion_path)
            except Exception as e:
                logger.warning("Could not persist final champion model: %s", e)

        logger.info("=================================================================")
        return {
            "total_runs": len(completed_entries),
            "best_run": best_overall,
            "all_runs": completed_entries,
        }

    def _simulate_telemetry(self, exp_id: int, decision: SupervisorDecision) -> ExperimentTelemetry:
        """Simulates realistic telemetry progression for loop testing before live model loading."""
        # Realistic progression: baseline starts ~0.842, climbs to ~0.868 as hyperparameters and features refine
        base_auc = 0.8420
        auc_progression = [
            0.8420, 0.8455, 0.8480, 0.8525, 0.8560, 0.8590, 0.8610,
            0.8625, 0.8640, 0.8652, 0.8660, 0.8668, 0.8675, 0.8680, 0.8685
        ]
        auc = auc_progression[exp_id - 1] if exp_id <= len(auc_progression) else base_auc + (exp_id * 0.0018)
        pr_auc = auc * 0.78  # Typical ratio for Telco Churn

        # Losses
        train_loss = 0.35 - (exp_id * 0.006)
        val_loss = 0.39 - (exp_id * 0.004)

        return ExperimentTelemetry(
            experiment_id=exp_id,
            oof_roc_auc=float(auc),
            oof_pr_auc=float(pr_auc),
            oof_log_loss=float(val_loss),
            oof_brier_score=0.125 - (exp_id * 0.001),
            fold_roc_aucs=[auc - 0.004, auc + 0.003, auc - 0.001, auc + 0.002, auc],
            final_train_loss=float(train_loss),
            final_val_loss=float(val_loss),
        )
