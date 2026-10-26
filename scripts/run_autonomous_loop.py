"""CLI entry point to execute the 15-experiment autonomous data scientist loop."""

import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.loop import AutonomousExperimentLoop
from src.data.loader import DataLoader
from src.supervisor.schemas import SupervisorDecision, HyperparameterConfig, FeatureEngineeringConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AADS-Runner")


def main():
    load_dotenv()
    logger.info("Initializing Autonomous AI Data Scientist (Utkrishta 2026 Challenge 1)...")

    orchestrator = AutonomousExperimentLoop()
    loader = DataLoader()

    try:
        train_val_df = loader.load_train_val()
    except Exception as e:
        logger.warning("Could not load processed dataset (%s). Check data/processed/ or run data preparation.", e)
        return

    # Run Step 0: Pre-Flight Profiling (Zero cost)
    locked_classifier = orchestrator.run_pre_flight(train_val_df)
    logger.info("Starting iterative optimization for locked classifier: %s", locked_classifier)

    # Initial Run 1 Config
    current_decision = SupervisorDecision(
        experiment_id=1,
        hypothesis="Run 1: Default baseline anchor with unaugmented features.",
        mathematical_justification="Establishing performance baseline on raw distributions.",
        hyperparameters=HyperparameterConfig(model_family="catboost", depth=6, learning_rate=0.05, l2_leaf_reg=3.0),
        feature_config=FeatureEngineeringConfig(),
        expected_outcome="Anchor baseline ROC-AUC around ~0.84.",
    )

    for run_idx in range(1, 16):
        if orchestrator.budget_guard.is_exhausted():
            logger.info("Budget exhausted (15/15 runs completed). Halting loop.")
            break

        logger.info("=== Starting Experiment %d of 15 ===", run_idx)
        result = orchestrator.run_experiment(run_idx, current_decision, train_val_df)

        if run_idx < 15:
            # Generate next decision via Math Engine + LLM Supervisor
            best_run = orchestrator.ledger.get_best_run()
            best_auc = best_run["metrics"]["oof_roc_auc"] if best_run else result["metrics"]["oof_roc_auc"]
            diagnosis = orchestrator.math_engine.analyze(result["metrics"], best_telemetry=best_run["metrics"] if best_run else None)
            prompt = orchestrator.prompt_builder.build_prompt(
                experiment_id=run_idx + 1,
                history=orchestrator.ledger.load_all_runs(),
                current_telemetry=result["metrics"],
                math_diagnosis=diagnosis,
                best_auc=best_auc,
            )
            current_decision = orchestrator.llm_client.decide_next_experiment(prompt, experiment_id=run_idx + 1)


if __name__ == "__main__":
    main()
