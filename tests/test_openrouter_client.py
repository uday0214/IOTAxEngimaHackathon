"""Unit tests for OpenRouter LLM supervisor integration and JSON extraction."""

import json
import os
from unittest.mock import MagicMock, patch
import httpx
import pytest

from src.supervisor.llm_client import (
    SupervisorLLMClient,
    GeminiSupervisorClient,
    clean_and_extract_json,
)
from src.supervisor.schemas import SupervisorDecision


SAMPLE_DECISION_PAYLOAD = {
    "experiment_id": 1,
    "hypothesis": "Test OpenRouter hypothesis with regularized tree depth.",
    "mathematical_justification": "Generalization gap minimization.",
    "hyperparameters": {
        "model_family": "catboost",
        "depth": 5,
        "l2_leaf_reg": 5.0,
        "learning_rate": 0.03,
        "iterations": 1200,
        "bagging_temperature": 0.3,
        "class_weight_mode": "None",
    },
    "feature_config": {
        "enable_charge_ratios": True,
        "enable_tenure_cohorts": True,
        "enable_service_density": True,
        "enable_interaction_terms": False,
        "drop_low_importance_features": [],
    },
    "expected_outcome": "Improved ROC-AUC by +0.004.",
}


def test_clean_and_extract_json():
    # 1. Clean JSON string
    raw = json.dumps(SAMPLE_DECISION_PAYLOAD)
    assert clean_and_extract_json(raw) == raw

    # 2. Markdown fenced JSON
    fenced = f"```json\n{raw}\n```"
    assert clean_and_extract_json(fenced) == raw

    # 3. Conversational wrapper text around JSON
    wrapped = f"Here is the next configuration:\n```json\n{raw}\n```\nHope this helps!"
    extracted = clean_and_extract_json(wrapped)
    parsed = json.loads(extracted)
    assert parsed["hyperparameters"]["depth"] == 5


def test_openrouter_client_initialization_auto():
    # When OPENROUTER_API_KEY is present
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key-123", "MOCK_SUPERVISOR": "false"}):
        client = SupervisorLLMClient()
        assert client.provider == "openrouter"
        assert client.api_key == "sk-or-test-key-123"
        assert not client.mock_mode


def test_openrouter_client_mock_fallback():
    # When no keys are present, falls back to mock mode
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "GEMINI_API_KEY": "", "LLM_PROVIDER": ""}):
        client = SupervisorLLMClient()
        assert client.provider == "mock"
        assert client.mock_mode is True

        decision = client.decide_next_experiment("dummy prompt", experiment_id=2)
        assert isinstance(decision, SupervisorDecision)
        assert decision.experiment_id == 2


def test_openrouter_api_call_success():
    client = SupervisorLLMClient(
        provider="openrouter",
        api_key="sk-or-live-key-xyz",
        model_name="google/gemini-2.5-flash",
    )
    assert not client.mock_mode

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(SAMPLE_DECISION_PAYLOAD),
                }
            }
        ]
    }
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        decision = client.decide_next_experiment("Telemetry: loss=0.38, auc=0.84", experiment_id=3)

        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-or-live-key-xyz"
        assert call_kwargs["json"]["model"] == "google/gemini-2.5-flash"
        assert call_kwargs["json"]["response_format"] == {"type": "json_object"}

        assert isinstance(decision, SupervisorDecision)
        assert decision.experiment_id == 3  # Ensures experiment_id is aligned
        assert decision.hyperparameters.depth == 5
        assert decision.hyperparameters.l2_leaf_reg == 5.0


def test_openrouter_api_retry_and_mock_fallback():
    client = SupervisorLLMClient(
        provider="openrouter",
        api_key="sk-or-failing-key",
        max_retries=2,
        base_retry_delay=0.01,
    )

    with patch("httpx.Client.post", side_effect=httpx.HTTPStatusError("503 Service Unavailable", request=MagicMock(), response=MagicMock())):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            decision = client.decide_next_experiment("Prompt", experiment_id=5)

            # Falls back gracefully to mock decision without crashing
            assert isinstance(decision, SupervisorDecision)
            assert decision.experiment_id == 5


def test_gemini_supervisor_client_backward_compatibility():
    # Instantiating GeminiSupervisorClient works identically
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-compat-key", "MOCK_SUPERVISOR": "false"}):
        client = GeminiSupervisorClient()
        assert client.provider == "openrouter"
        assert client.api_key == "sk-or-compat-key"


def test_generate_post_training_report_mock():
    client = SupervisorLLMClient(provider="mock")
    best_run = {
        "experiment_id": 14,
        "hyperparameters": {"depth": 5, "learning_rate": 0.02, "l2_leaf_reg": 12.0, "iterations": 1500},
        "feature_config": {"enable_interaction_terms": True, "drop_low_importance_features": ["gender"]},
        "metrics": {"oof_roc_auc": 0.8685, "oof_pr_auc": 0.698, "oof_log_loss": 0.358, "overfitting_index": 4.47, "ece_score": 0.019},
    }
    baseline_run = {
        "experiment_id": 1,
        "metrics": {"oof_roc_auc": 0.842, "oof_pr_auc": 0.6568, "oof_log_loss": 0.412, "overfitting_index": 10.19},
    }
    summary_stats = {
        "net_improvements": {
            "delta_roc_auc": 0.0265,
            "pct_gain_roc_auc": 3.15,
            "delta_pr_auc": 0.0412,
            "pct_gain_pr_auc": 6.27,
            "delta_log_loss": -0.054,
            "delta_overfitting_index": -5.72,
        }
    }

    report = client.generate_post_training_report(
        all_runs=[baseline_run, best_run],
        best_run=best_run,
        baseline_run=baseline_run,
        summary_stats=summary_stats,
    )

    assert isinstance(report, str)
    assert "# Autonomous AI Data Scientist: 15-Experiment Post-Mortem Report" in report
    assert "Executive Mathematical Scorecard" in report
    assert "0.8685" in report


def test_verify_connection():
    # 1. Mock mode
    mock_client = SupervisorLLMClient(provider="mock")
    res_mock = mock_client.verify_connection()
    assert res_mock["status"] == "MOCK_MODE"

    # 2. OpenRouter with custom model
    or_client = SupervisorLLMClient(
        provider="openrouter",
        api_key="sk-or-test-verification",
        model_name="nvidia/nemotron-3-ultra-550b-a55b:free",
    )
    assert or_client.model_name == "nvidia/nemotron-3-ultra-550b-a55b:free"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": '{"status": "ready"}'}}]}
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_resp):
        res_or = or_client.verify_connection()
        assert res_or["status"] == "READY"
        assert res_or["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
        assert res_or["latency_sec"] >= 0.0
