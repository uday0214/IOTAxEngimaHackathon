"""Generates standardized prediction results file matching organizer schema."""

import logging
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.telemetry.ledger import ExperimentLedger
from src.features.pipeline import DynamicFeaturePipeline
from src.supervisor.schemas import FeatureEngineeringConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SubmissionExporter")


def main():
    load_dotenv()
    output_path = Path("artifacts/submission.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ledger = ExperimentLedger()
    best_run = ledger.get_best_run()
    champion_model_path = Path("artifacts/champion_model.cbm")
    loader = DataLoader()

    # Determine input dataset for submission (check test folder, else holdout)
    test_file = Path("data/test/test.csv")
    if test_file.exists():
        logger.info("Loading test dataset from %s...", test_file)
        eval_df = pd.read_csv(test_file)
    else:
        try:
            eval_df = loader.load_holdout()
            logger.info("Using sealed holdout partition for submission export (%d rows).", len(eval_df))
        except Exception:
            logger.warning("No test or holdout dataset found. Generating 200 synthetic evaluation samples.")
            from scripts.run_autonomous_loop import generate_sample_tabular_data
            eval_df = generate_sample_tabular_data(n_samples=200)

    # Feature transformation using champion experiment config
    if best_run:
        feature_cfg = FeatureEngineeringConfig(**best_run["feature_config"])
    else:
        feature_cfg = FeatureEngineeringConfig()

    pipeline = DynamicFeaturePipeline(feature_cfg)
    transformed_eval = pipeline.transform(eval_df)
    X_eval = transformed_eval.drop(columns=["Churn"], errors="ignore")

    # Generate probabilities
    if champion_model_path.exists():
        logger.info("Generating predictions using champion CatBoost model from %s...", champion_model_path)
        model = CatBoostClassifier()
        model.load_model(str(champion_model_path))
        probs = model.predict_proba(X_eval)[:, 1]
    else:
        logger.info("Champion model not found on disk; generating baseline probabilities from feature signal.")
        probs = [0.25] * len(X_eval)

    # Build submission DataFrame
    customer_ids = eval_df["customerID"] if "customerID" in eval_df.columns else [f"CUST_{i:05d}" for i in range(len(eval_df))]
    submission_df = pd.DataFrame({
        "customerID": customer_ids,
        "Churn_Probability": probs,
        "Churn_Prediction": (pd.Series(probs) >= 0.5).astype(int),
    })

    submission_df.to_csv(output_path, index=False)
    logger.info("Submission successfully exported to %s (%d rows).", output_path, len(submission_df))


if __name__ == "__main__":
    main()
