"""Unit tests for ExtensiveTelemetryCollector and telemetry persistence."""

import json
from pathlib import Path
import numpy as np
from src.telemetry.collector import ExtensiveTelemetryCollector


def test_classification_diagnostics_calculation():
    # Synthetic labels & probabilities
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    diag = ExtensiveTelemetryCollector.compute_classification_diagnostics(y_true, y_prob)

    assert "oof_roc_auc" in diag
    assert diag["oof_roc_auc"] == 1.0  # Perfectly ordered
    assert "oof_pr_auc" in diag
    assert "ece_score" in diag
    assert diag["ece_score"] >= 0.0
    assert "optimal_youden_threshold" in diag
    assert 0.4 <= diag["optimal_youden_threshold"] <= 0.6
    assert "confusion_matrix_05" in diag
    assert "false_negative_rate" in diag


def test_feature_importance_spectrum_sorting():
    importances = {
        "Contract": 45.2,
        "tenure": 30.1,
        "MonthlyCharges": 15.0,
        "gender": 0.0,
        "PhoneService": 0.0,
    }
    top_f, bottom_f, zero_f = ExtensiveTelemetryCollector.process_feature_importance_spectrum(importances)

    assert top_f[0]["feature"] == "Contract"
    assert "gender" in zero_f
    assert "PhoneService" in zero_f


def test_loss_trajectory_sampling():
    train_losses = [float(0.5 - i * 0.01) for i in range(50)]
    val_losses = [float(0.48 - i * 0.005) for i in range(50)]

    sampled, tail_slope = ExtensiveTelemetryCollector.sample_loss_trajectories(train_losses, val_losses, n_samples=10)

    assert len(sampled) == 10
    assert sampled[0]["iteration"] == 0
    assert sampled[-1]["iteration"] == 49
    assert tail_slope < 0.0  # Loss was descending


def test_telemetry_artifact_saving(tmp_path: Path):
    telemetry_data = {
        "experiment_id": 1,
        "oof_roc_auc": 0.852,
        "oof_pr_auc": 0.672,
    }
    file_path = ExtensiveTelemetryCollector.save_detailed_telemetry(
        experiment_id=1,
        telemetry_dict=telemetry_data,
        output_dir=tmp_path,
    )
    assert file_path.exists()
    with open(file_path, "r") as f:
        loaded = json.load(f)
    assert loaded["experiment_id"] == 1
    assert loaded["oof_roc_auc"] == 0.852
