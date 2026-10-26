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
