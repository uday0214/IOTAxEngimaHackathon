"""Google AI Studio Gemini API client wrapper for Supervisor agent."""

import os
import logging
from typing import Optional
from src.supervisor.schemas import SupervisorDecision, HyperparameterConfig, FeatureEngineeringConfig
from src.supervisor.prompt_builder import SupervisorPromptBuilder

logger = logging.getLogger(__name__)


class GeminiSupervisorClient:
    """Invokes Google Gemini with structured Pydantic schemas or mocks offline."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.mock_mode = os.getenv("MOCK_SUPERVISOR", "false").lower() == "true" or not self.api_key

    def decide_next_experiment(self, prompt: str, experiment_id: int) -> SupervisorDecision:
        """Emits structured SupervisorDecision using Gemini or mock fallbacks."""
        if self.mock_mode:
            logger.info("Running supervisor in MOCK_MODE (No Gemini API call).")
            return self._mock_decision(experiment_id)

        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "system_instruction": SupervisorPromptBuilder.SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "response_schema": SupervisorDecision,
                    "temperature": 0.2,
                },
            )
            return SupervisorDecision.model_validate_json(response.text)
        except Exception as e:
            logger.error("Error communicating with Gemini API: %s. Falling back to mock decision.", e)
            return self._mock_decision(experiment_id)

    def _mock_decision(self, experiment_id: int) -> SupervisorDecision:
        # Dynamic progression for testing
        depth = 4 if experiment_id > 3 else 6
        l2 = 5.0 + (experiment_id * 0.5)
        lr = 0.03 if experiment_id > 6 else 0.05
        return SupervisorDecision(
            experiment_id=experiment_id,
            hypothesis=f"Cycle {experiment_id}: Regularizing depth to {depth} and tuning L2 to {l2:.1f}.",
            mathematical_justification="Overfitting index indicates need for increased tree regularization.",
            hyperparameters=HyperparameterConfig(
                model_family="catboost",
                depth=depth,
                l2_leaf_reg=l2,
                learning_rate=lr,
                iterations=1200,
                bagging_temperature=0.3,
            ),
            feature_config=FeatureEngineeringConfig(
                enable_charge_ratios=True,
                enable_tenure_cohorts=True,
                enable_service_density=True,
                enable_interaction_terms=(experiment_id > 5),
            ),
            expected_outcome="Reduce generalization gap by 2.5% and improve OOF ROC-AUC.",
        )
