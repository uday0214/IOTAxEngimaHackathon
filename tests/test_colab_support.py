"""Unit tests for Google Colab support, hardware acceleration, and environment resolution."""

import os
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

from src.utils.environment import (
    is_google_colab,
    is_gpu_available,
    get_device,
    get_gemini_api_key,
    get_system_info,
)
from src.supervisor.schemas import HyperparameterConfig
from src.models.catboost_model import CatBoostPipelineModel


def test_is_google_colab_detection():
    # Outside Colab
    with patch.dict("sys.modules", {"google.colab": MagicMock()}):
        assert is_google_colab() is True

    with patch.dict("sys.modules", {}, clear=False):
        if "google.colab" in os.sys.modules:
            del os.sys.modules["google.colab"]
        # In Linux container without /content
        with patch("os.path.exists", return_value=False):
            assert is_google_colab() is False


def test_get_device_resolution():
    # CPU requested
    assert get_device("cpu") == "CPU"
    assert get_device("CPU") == "CPU"

    # Auto when GPU unavailable
    with patch("src.utils.environment.is_gpu_available", return_value=False):
        assert get_device("auto") == "CPU"
        assert get_device("gpu") == "CPU"

    # Auto when GPU available
    with patch("src.utils.environment.is_gpu_available", return_value=True):
        assert get_device("auto") == "GPU"
        assert get_device("gpu") == "GPU"


def test_get_gemini_api_key_from_env():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-env-key-123"}):
        assert get_gemini_api_key() == "test-env-key-123"


def test_get_gemini_api_key_from_colab_userdata():
    mock_userdata = MagicMock()
    mock_userdata.get.return_value = "colab-secret-key-456"

    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
        with patch("src.utils.environment.is_google_colab", return_value=True):
            with patch.dict("sys.modules", {"google.colab": MagicMock(userdata=mock_userdata), "google.colab.userdata": mock_userdata}):
                with patch("src.utils.environment.get_colab_secret", return_value="colab-secret-key-456"):
                    assert get_gemini_api_key() == "colab-secret-key-456"


def test_catboost_device_initialization_and_fit():
    config = HyperparameterConfig(iterations=300, depth=4)
    model = CatBoostPipelineModel(config=config, experiment_id=1, device="cpu")
    assert model.device == "CPU"

    X = pd.DataFrame({
        "num1": np.random.randn(30),
        "cat1": ["A", "B", "C"] * 10,
    })
    y = pd.Series(np.random.randint(0, 2, 30))

    telemetry = model.cross_validate(X, y, n_splits=3)
    assert telemetry.experiment_id == 1
    assert 0.0 <= telemetry.oof_roc_auc <= 1.0
    assert len(telemetry.fold_roc_aucs) == 3


def test_catboost_gpu_fallback_to_cpu():
    config = HyperparameterConfig(iterations=300, depth=4)
    # Force GPU mode on CPU machine to trigger graceful fallback
    model = CatBoostPipelineModel(config=config, experiment_id=1, device="cpu")
    model.device = "GPU"

    X = pd.DataFrame({
        "num1": np.random.randn(30),
        "cat1": ["A", "B", "C"] * 10,
    })
    y = pd.Series(np.random.randint(0, 2, 30))

    # Fitting should catch CatBoostError / CUDA error and fall back to CPU seamlessly
    telemetry = model.cross_validate(X, y, n_splits=3)
    assert model.device == "CPU"
    assert telemetry.experiment_id == 1
