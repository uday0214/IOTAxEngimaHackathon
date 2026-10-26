"""CLI entry point to execute the 15-experiment autonomous data scientist loop.

Supports both:
  - Dry-run / Simulation mode: Runs the complete end-to-end supervisor loop without loading
    heavy model weights.
  - Live mode (--live): Runs live CatBoost 5-fold cross-validation when ready.
"""

import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.loop import AutonomousExperimentLoop
from src.data.loader import DataLoader
from src.utils.environment import is_google_colab, get_system_info, get_device, setup_colab_drive
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AADS-Runner")


def generate_sample_tabular_data(n_samples: int = 500) -> pd.DataFrame:
    """Generates sample tabular churn data for loop verification if dataset not downloaded yet."""
    import numpy as np

    np.random.seed(42)
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    payment = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_samples)
    tenure = np.random.randint(0, 72, size=n_samples)
    monthly = np.random.uniform(18.0, 118.0, size=n_samples)
    total = tenure * monthly + np.random.normal(0, 10, size=n_samples)
    total = np.maximum(total, 0.0)

    # Synthetic churn target correlated with contract & tenure
    prob = 0.2 + 0.3 * (contracts == "Month-to-month") - 0.2 * (tenure > 30)
    prob = np.clip(prob, 0.05, 0.95)
    churn = (np.random.rand(n_samples) < prob).astype(int)

    return pd.DataFrame({
        "gender": np.random.choice(["Male", "Female"], size=n_samples),
        "SeniorCitizen": np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16]),
        "Partner": np.random.choice(["Yes", "No"], size=n_samples),
        "Dependents": np.random.choice(["Yes", "No"], size=n_samples),
        "tenure": tenure,
        "PhoneService": np.random.choice(["Yes", "No"], size=n_samples),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], size=n_samples),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
        "Contract": contracts,
        "PaperlessBilling": np.random.choice(["Yes", "No"], size=n_samples),
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": churn,
    })


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Autonomous AI Data Scientist (Utkrishta 2026 Challenge 1)")
    parser.add_argument("--live", action="store_true", help="Execute live model fitting (loads CatBoost)")
    parser.add_argument("--max-runs", type=int, default=15, help="Maximum experiment limit (default: 15)")
    parser.add_argument("--clean-ledger", action="store_true", help="Clear previous ledger runs before starting")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Hardware execution device target: auto, cpu, or gpu (default: auto)",
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Enable Google Colab environment integration (auto-mount Google Drive if available)",
    )
    args = parser.parse_args()

    # Environmental Telemetry & Google Colab Integration
    sys_info = get_system_info()
    colab_active = is_google_colab() or args.colab
    active_device = get_device(args.device)

    logger.info("=================================================================")
    logger.info("ENVIRONMENT & RUNTIME CONFIGURATION:")
    logger.info("  Google Colab Runtime : %s", "DETECTED / ACTIVE" if colab_active else "LOCAL / STANDARD")
    logger.info("  Requested Device     : %s | Resolved Target: %s", args.device.upper(), active_device)
    logger.info("  GPU Acceleration     : %s (%s)", sys_info["is_gpu_available"], sys_info["gpu_name"])
    logger.info("  Python Version       : %s on %s", sys_info["python_version"], sys_info["platform"])
    logger.info("=================================================================")

    if colab_active and args.colab:
        setup_colab_drive()

    ledger_path = Path("artifacts/experiment_ledger.jsonl")
    if args.clean_ledger and ledger_path.exists():
        ledger_path.unlink()
        logger.info("Cleared previous ledger at %s", ledger_path)

    logger.info(
        "System mode: %s (Execute live model: %s, Device: %s)",
        "LIVE TRAINING" if args.live else "LOOP VERIFICATION (Simulated Telemetry)",
        args.live,
        active_device,
    )

    # Initialize Orchestrator Loop
    orchestrator = AutonomousExperimentLoop(
        max_runs=args.max_runs,
        ledger_path=ledger_path,
        execute_live_training=args.live,
        device=active_device,
    )

    # Load canonical or sample dataset
    loader = DataLoader()
    try:
        train_val_df = loader.load_train_val()
        logger.info("Successfully loaded canonical dataset partition (%d rows).", len(train_val_df))
    except Exception:
        logger.info("Canonical processed dataset not found on disk yet. Generating verified schema partition (%d rows) for loop verification.", 600)
        train_val_df = generate_sample_tabular_data(n_samples=600)

    # Execute the full 15-run loop
    results = orchestrator.execute_full_15_run_loop(train_val_df)
    logger.info("Execution complete. Total experiments recorded: %d", results["total_runs"])


if __name__ == "__main__":
    main()
