"""Unit tests for the Mathematical Inference Engine diagnostics."""

import numpy as np
from src.supervisor.math_engine import MathematicalInferenceEngine


def test_overfitting_index_calculation():
    engine = MathematicalInferenceEngine()
    gap, idx = engine.compute_overfitting_index(train_loss=0.30, val_loss=0.40)
    assert np.isclose(gap, 0.10)
    assert np.isclose(idx, 25.0)  # 0.10 / 0.40 * 100 = 25%


def test_youden_threshold_maximization():
    engine = MathematicalInferenceEngine()
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    t = engine.compute_youden_threshold(y_true, y_prob)
    assert 0.4 <= t <= 0.6


def test_ece_calibration():
    engine = MathematicalInferenceEngine()
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    ece = engine.compute_ece(y_true, y_prob)
    assert ece >= 0.0


def test_math_diagnosis_regimes_and_directives():
    engine = MathematicalInferenceEngine()

    # Exp 1 (leading into Exp 2 -> Phase 1)
    diag1 = engine.analyze(current_telemetry={"experiment_id": 1, "final_train_loss": 0.35, "final_val_loss": 0.38, "oof_roc_auc": 0.842})
    assert "PHASE 1" in diag1.regime_phase
    assert diag1.suggested_depth_bounds == (4, 7)

    # Exp 5 (leading into Exp 6 -> Phase 2)
    diag2 = engine.analyze(current_telemetry={"experiment_id": 5, "final_train_loss": 0.32, "final_val_loss": 0.36, "oof_roc_auc": 0.855})
    assert "PHASE 2" in diag2.regime_phase

    # Exp 9 (leading into Exp 10 -> Phase 3)
    diag3 = engine.analyze(current_telemetry={"experiment_id": 9, "final_train_loss": 0.30, "final_val_loss": 0.34, "oof_roc_auc": 0.862})
    assert "PHASE 3" in diag3.regime_phase

    # Exp 13 (leading into Exp 14 -> Phase 4)
    diag4 = engine.analyze(current_telemetry={"experiment_id": 13, "final_train_loss": 0.28, "final_val_loss": 0.33, "oof_roc_auc": 0.867})
    assert "PHASE 4" in diag4.regime_phase
    assert diag4.recommended_class_weight == "Balanced"
    assert len(diag4.mathematical_rationale_bullet_points) > 0
