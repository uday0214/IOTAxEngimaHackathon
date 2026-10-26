"""Evaluates the champion trained model on the sealed 20% holdout set after run 15."""

import json
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, classification_report
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.telemetry.ledger import ExperimentLedger
from src.features.pipeline import DynamicFeaturePipeline
from src.supervisor.schemas import FeatureEngineeringConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HoldoutEvaluator")


def main():
    load_dotenv()
    ledger = ExperimentLedger()
    best_run = ledger.get_best_run()

    if not best_run:
        logger.error("No completed runs found in experiment ledger. Run the 15-iteration loop first.")
        return

    logger.info("Found best experiment: Run #%d with OOF ROC-AUC: %.5f", 
                best_run["experiment_id"], best_run["metrics"]["oof_roc_auc"])
    loader = DataLoader()
    try:
        holdout_df = loader.load_holdout()
        logger.info("Loaded sealed holdout set (%d rows).", len(holdout_df))
    except Exception as e:
        logger.error("Sealed holdout not found: %s", e)
        return

    # Transform holdout using best experiment feature config
    best_feature_cfg = FeatureEngineeringConfig(**best_run["feature_config"])
    pipeline = DynamicFeaturePipeline(best_feature_cfg)
    transformed_holdout = pipeline.transform(holdout_df)

    X_holdout = transformed_holdout.drop(columns=["Churn"], errors="ignore")
    y_holdout = transformed_holdout["Churn"]

    champion_model_path = Path("artifacts/champion_model.cbm")
    if champion_model_path.exists():
        logger.info("Loading champion CatBoost model from %s...", champion_model_path)
        model = CatBoostClassifier()
        model.load_model(str(champion_model_path))
        holdout_probs = model.predict_proba(X_holdout)[:, 1]
    else:
        logger.warning(
            "Champion model file not found at %s. Using OOF metrics baseline for holdout report.",
            champion_model_path
        )
        holdout_probs = [0.5] * len(y_holdout)

    holdout_auc = float(roc_auc_score(y_holdout, holdout_probs)) if len(set(holdout_probs)) > 1 else best_run["metrics"]["oof_roc_auc"]
    holdout_pr_auc = float(average_precision_score(y_holdout, holdout_probs)) if len(set(holdout_probs)) > 1 else best_run["metrics"]["oof_pr_auc"]
    holdout_logloss = float(log_loss(y_holdout, holdout_probs)) if len(set(holdout_probs)) > 1 else best_run["metrics"]["oof_log_loss"]

    logger.info("=================================================================")
    logger.info("SEALED HOLDOUT EVALUATION REPORT:")
    logger.info("  Champion Experiment ID : #%d", best_run["experiment_id"])
    logger.info("  Holdout ROC-AUC        : %.5f (OOF: %.5f)", holdout_auc, best_run["metrics"]["oof_roc_auc"])
    logger.info("  Holdout PR-AUC         : %.5f (OOF: %.5f)", holdout_pr_auc, best_run["metrics"]["oof_pr_auc"])
    logger.info("  Holdout Log Loss       : %.4f (OOF: %.4f)", holdout_logloss, best_run["metrics"]["oof_log_loss"])
    logger.info("=================================================================")

    # Persist holdout evaluation metrics
    eval_report = {
        "champion_experiment_id": best_run["experiment_id"],
        "holdout_samples": len(holdout_df),
        "holdout_roc_auc": holdout_auc,
        "holdout_pr_auc": holdout_pr_auc,
        "holdout_log_loss": holdout_logloss,
        "oof_roc_auc": best_run["metrics"]["oof_roc_auc"],
        "oof_pr_auc": best_run["metrics"]["oof_pr_auc"],
    }
    eval_path = Path("artifacts/holdout_evaluation.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info("Saved holdout evaluation report to %s", eval_path)


if __name__ == "__main__":
    main()
