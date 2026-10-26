"""Evaluates the best trained model on the sealed 20% holdout set after run 15."""

import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.telemetry.ledger import ExperimentLedger

logging.basicConfig(level=logging.INFO)
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
        logger.info("Loaded sealed holdout set (%d rows). Ready for final evaluation.", len(holdout_df))
    except Exception as e:
        logger.warning("Sealed holdout not found: %s", e)


if __name__ == "__main__":
    main()
