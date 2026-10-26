"""Google AI Studio Gemini API client wrapper for Supervisor agent.

Provides structured schema-constrained prompting, exponential backoff retries,
and seamless fallback to offline mock mode for CI/testing.
"""

import os
import time
import json
import logging
from typing import Optional
from pydantic import ValidationError

from src.supervisor.schemas import (
    SupervisorDecision,
    HyperparameterConfig,
    FeatureEngineeringConfig,
)
from src.supervisor.prompt_builder import SupervisorPromptBuilder
from src.utils.environment import get_gemini_api_key

logger = logging.getLogger(__name__)


class GeminiSupervisorClient:
    """Client for Google AI Studio Gemini API acting as the experiment supervisor."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 3,
        base_retry_delay: float = 2.0,
    ):
        self.api_key = api_key or get_gemini_api_key()
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay

        # Mock mode if explicitly requested or if no API key is provided
        self.mock_mode = (
            os.getenv("MOCK_SUPERVISOR", "false").lower() == "true"
            or not self.api_key
            or self.api_key == "your_gemini_api_key_here"
        )

        if self.mock_mode:
            logger.info("GeminiSupervisorClient initialized in MOCK MODE (No external API calls will be made).")
        else:
            logger.info("GeminiSupervisorClient initialized with model: %s", self.model_name)

    def decide_next_experiment(self, prompt: str, experiment_id: int) -> SupervisorDecision:
        """Invokes Gemini to emit a typed SupervisorDecision, with retry backoff."""
        if self.mock_mode:
            return self._mock_decision(experiment_id)

        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw_json = self._call_gemini_api(prompt)
                decision = SupervisorDecision.model_validate_json(raw_json)
                # Enforce consistent experiment_id
                decision.experiment_id = experiment_id
                logger.info("Successfully received Supervisor decision for Run %d from Gemini API.", experiment_id)
                return decision
            except (ValidationError, json.JSONDecodeError) as e:
                logger.warning(
                    "Attempt %d/%d: Failed to parse Gemini response as SupervisorDecision schema (%s).",
                    attempt, self.max_retries, e
                )
                last_exception = e
            except Exception as e:
                logger.warning(
                    "Attempt %d/%d: Gemini API call failed with error: %s",
                    attempt, self.max_retries, e
                )
                last_exception = e

            if attempt < self.max_retries:
                sleep_sec = self.base_retry_delay * (2 ** (attempt - 1))
                logger.info("Retrying Gemini API call in %.1f seconds...", sleep_sec)
                time.sleep(sleep_sec)

        logger.error(
            "All %d Gemini API attempts failed (%s). Falling back to deterministic mock decision.",
            self.max_retries, last_exception
        )
        return self._mock_decision(experiment_id)

    def _call_gemini_api(self, prompt: str) -> str:
        """Calls Google AI Studio API using either google-genai or google.generativeai."""
        # Method 1: New official Google GenAI SDK (google-genai)
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SupervisorPromptBuilder.SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=SupervisorDecision,
                    temperature=0.2,
                ),
            )
            if response.text:
                return response.text
        except ImportError:
            pass

        # Method 2: Legacy google.generativeai SDK
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=SupervisorPromptBuilder.SYSTEM_PROMPT,
                generation_config={"response_mime_type": "application/json", "temperature": 0.2},
            )
            response = model.generate_content(prompt)
            if response.text:
                return response.text
        except ImportError:
            pass

        raise RuntimeError(
            "Neither 'google-genai' nor 'google-generativeai' is installed. "
            "Please install dependencies via `pip install -r requirements.txt`."
        )

    def _mock_decision(self, experiment_id: int) -> SupervisorDecision:
        """Deterministic, progression-based decision for offline testing and simulations."""
        # Rational parameter progression across 15 runs
        if experiment_id <= 3:
            depth = 6
            l2 = 3.0 + (experiment_id * 1.0)
            lr = 0.05
            enable_interactions = False
            drop_features = []
            hyp = f"Run {experiment_id}: Capacity exploration with depth={depth} and initial L2 leaf reg={l2}."
            math_just = "Establishing benchmark and measuring initial generalization gap."
        elif experiment_id <= 7:
            depth = 4  # Regularize depth to combat overfitting
            l2 = 6.0 + (experiment_id * 1.5)
            lr = 0.04
            enable_interactions = (experiment_id >= 5)
            drop_features = ["gender", "PhoneService"] if experiment_id >= 6 else []
            hyp = f"Run {experiment_id}: Reducing tree depth to {depth} and pruning low-signal features."
            math_just = "Overfitting index Omega > 15% indicates need for structural tree regularization."
        elif experiment_id <= 11:
            depth = 5
            l2 = 10.0
            lr = 0.025  # Learning rate decay with high iteration count
            enable_interactions = True
            drop_features = ["gender", "PhoneService"]
            hyp = f"Run {experiment_id}: Learning rate decay (lr={lr}) with Bayesian bootstrap bagging."
            math_just = "Fine-tuning convergence dynamics around optimal validation loss plateau."
        else:
            depth = 4
            l2 = 12.0
            lr = 0.02
            enable_interactions = True
            drop_features = ["gender", "PhoneService"]
            hyp = f"Run {experiment_id}: Final hyper-refinement with class weight balancing."
            math_just = "Optimizing Expected Calibration Error and Youden threshold for PR-AUC gain."

        return SupervisorDecision(
            experiment_id=experiment_id,
            hypothesis=hyp,
            mathematical_justification=math_just,
            hyperparameters=HyperparameterConfig(
                model_family="catboost",
                depth=depth,
                l2_leaf_reg=l2,
                learning_rate=lr,
                iterations=1000 + (experiment_id * 100),
                bagging_temperature=0.2 + (experiment_id * 0.04),
                class_weight_mode="Balanced" if experiment_id >= 10 else "None",
            ),
            feature_config=FeatureEngineeringConfig(
                enable_charge_ratios=True,
                enable_tenure_cohorts=True,
                enable_service_density=True,
                enable_interaction_terms=enable_interactions,
                drop_low_importance_features=drop_features,
            ),
            expected_outcome=f"Targeting +0.003 ROC-AUC improvement and reduced generalization gap in cycle {experiment_id}.",
        )
