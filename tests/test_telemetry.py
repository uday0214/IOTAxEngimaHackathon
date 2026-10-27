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

    # Check new visualizable curve points and histogram
    assert "roc_curve_points" in diag
    assert len(diag["roc_curve_points"]) > 0
    assert "pr_curve_points" in diag
    assert len(diag["pr_curve_points"]) > 0
    assert "calibration_curve_points" in diag
    assert "probability_histogram" in diag
    assert len(diag["probability_histogram"]) == 10


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


def test_trajectory_summary_and_dashboard_generation(tmp_path: Path):
    from src.telemetry.visualization import generate_interactive_dashboard

    mock_entries = [
        {
            "experiment_id": 1,
            "hyperparameters": {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "iterations": 1000, "class_weight_mode": "None"},
            "feature_config": {"enable_interaction_terms": False, "drop_low_importance_features": []},
            "metrics": {
                "oof_roc_auc": 0.842,
                "oof_pr_auc": 0.650,
                "oof_log_loss": 0.410,
                "overfitting_index": 10.2,
                "generalization_gap": 0.040,
                "ece_score": 0.045,
                "optimal_youden_threshold": 0.38,
                "training_duration_seconds": 1.2,
            },
        },
        {
            "experiment_id": 2,
            "hyperparameters": {"depth": 5, "learning_rate": 0.04, "l2_leaf_reg": 5.0, "iterations": 1200, "class_weight_mode": "None"},
            "feature_config": {"enable_interaction_terms": True, "drop_low_importance_features": ["gender"]},
            "metrics": {
                "oof_roc_auc": 0.855,
                "oof_pr_auc": 0.672,
                "oof_log_loss": 0.385,
                "overfitting_index": 6.5,
                "generalization_gap": 0.025,
                "ece_score": 0.032,
                "optimal_youden_threshold": 0.38,
                "training_duration_seconds": 1.4,
            },
        },
    ]

    summary_file = ExtensiveTelemetryCollector.save_trajectory_summary(mock_entries, output_dir=tmp_path)
    assert summary_file.exists()

    with open(summary_file, encoding="utf-8") as f:
        summary_data = json.load(f)

    assert summary_data["total_experiments"] == 2
    assert summary_data["champion_experiment_id"] == 2
    assert summary_data["net_improvements"]["delta_roc_auc"] > 0
    assert len(summary_data["trajectory_time_series"]) == 2

    # Dashboard generation test
    dash_html = tmp_path / "dashboard.html"
    res_dash = generate_interactive_dashboard(summary_path=summary_file, output_html=dash_html)
    assert res_dash.exists()
    assert dash_html.stat().st_size > 500
