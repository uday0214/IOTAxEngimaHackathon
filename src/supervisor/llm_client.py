"""LLM client wrapper for the Supervisor agent supporting OpenRouter and Google Gemini APIs.

Provides structured schema-constrained prompting, exponential backoff retries,
automatic provider failover, and seamless fallback to offline mock mode for CI/testing.
"""

import os
import time
import json
import logging
from typing import Optional
import httpx
from pydantic import ValidationError

from src.supervisor.schemas import (
    SupervisorDecision,
    HyperparameterConfig,
    FeatureEngineeringConfig,
)
from src.supervisor.prompt_builder import SupervisorPromptBuilder
from src.utils.environment import (
    get_gemini_api_key,
    get_openrouter_api_key,
    get_llm_provider,
)

logger = logging.getLogger(__name__)


def clean_and_extract_json(raw_text: str) -> str:
    """Extracts and sanitizes a JSON payload from raw LLM output string."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Check if directly valid JSON
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Attempt to slice outermost curly brackets
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    return text


class SupervisorLLMClient:
    """Unified client supporting OpenRouter and Google AI Studio Gemini API."""

    def __init__(
        self,
        provider: str = "auto",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 3,
        base_retry_delay: float = 2.0,
    ):
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay

        # Resolve provider
        requested_provider = provider.lower() if provider else "auto"
        if requested_provider == "auto":
            if api_key and api_key.startswith("sk-or-"):
                self.provider = "openrouter"
            else:
                self.provider = get_llm_provider()
        else:
            self.provider = requested_provider

        # Resolve credentials and model based on provider
        if self.provider == "openrouter":
            self.api_key = api_key or get_openrouter_api_key()
            self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
            self.mock_mode = (
                os.getenv("MOCK_SUPERVISOR", "false").lower() == "true"
                or not self.api_key
                or self.api_key == "your_openrouter_api_key_here"
            )
            if self.mock_mode:
                logger.info("SupervisorLLMClient (OpenRouter) initialized in MOCK MODE (No external API calls).")
            else:
                logger.info(
                    "SupervisorLLMClient initialized with OpenRouter provider (Model: %s).",
                    self.model_name,
                )

        elif self.provider == "gemini":
            self.api_key = api_key or get_gemini_api_key()
            self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.mock_mode = (
                os.getenv("MOCK_SUPERVISOR", "false").lower() == "true"
                or not self.api_key
                or self.api_key == "your_gemini_api_key_here"
            )
            if self.mock_mode:
                logger.info("SupervisorLLMClient (Gemini) initialized in MOCK MODE (No external API calls).")
            else:
                logger.info(
                    "SupervisorLLMClient initialized with Google Gemini provider (Model: %s).",
                    self.model_name,
                )

        else:  # mock
            self.provider = "mock"
            self.api_key = None
            self.model_name = "mock"
            self.mock_mode = True
            logger.info("SupervisorLLMClient initialized in MOCK MODE.")

    def decide_next_experiment(self, prompt: str, experiment_id: int) -> SupervisorDecision:
        """Invokes the configured LLM provider to emit a typed SupervisorDecision."""
        if self.mock_mode:
            return self._mock_decision(experiment_id)

        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.provider == "openrouter":
                    raw_json = self._call_openrouter_api(prompt)
                elif self.provider == "gemini":
                    raw_json = self._call_gemini_api(prompt)
                else:
                    return self._mock_decision(experiment_id)

                clean_json = clean_and_extract_json(raw_json)
                decision = SupervisorDecision.model_validate_json(clean_json)
                # Enforce consistent experiment_id
                decision.experiment_id = experiment_id
                logger.info(
                    "Successfully received Supervisor decision for Run %d from %s (%s).",
                    experiment_id,
                    self.provider.upper(),
                    self.model_name,
                )
                return decision

            except (ValidationError, json.JSONDecodeError) as e:
                logger.warning(
                    "Attempt %d/%d: Failed to parse %s response as SupervisorDecision schema (%s).",
                    attempt,
                    self.max_retries,
                    self.provider,
                    e,
                )
                last_exception = e
            except Exception as e:
                logger.warning(
                    "Attempt %d/%d: %s API call failed with error: %s",
                    attempt,
                    self.max_retries,
                    self.provider,
                    e,
                )
                last_exception = e

            if attempt < self.max_retries:
                sleep_sec = self.base_retry_delay * (2 ** (attempt - 1))
                logger.info("Retrying %s API call in %.1f seconds...", self.provider, sleep_sec)
                time.sleep(sleep_sec)

        # Failover check: If openrouter failed, try gemini if available; if gemini failed, try openrouter if available
        fallback_decision = self._try_fallback(prompt, experiment_id)
        if fallback_decision:
            return fallback_decision

        logger.error(
            "All %d %s API attempts failed (%s). Falling back to deterministic mock decision.",
            self.max_retries,
            self.provider,
            last_exception,
        )
        return self._mock_decision(experiment_id)

    def _call_openrouter_api(self, prompt: str) -> str:
        """Calls OpenRouter Chat Completions endpoint using httpx."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/uday0214/IOTAxEngimaHackathon",
            "X-Title": "IOTAxEnigma Autonomous ML Orchestrator",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SupervisorPromptBuilder.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content

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

    def _try_fallback(self, prompt: str, experiment_id: int) -> Optional[SupervisorDecision]:
        """Attempts cross-provider fallback if secondary API key is available."""
        if self.provider == "openrouter":
            gemini_key = get_gemini_api_key()
            if gemini_key:
                logger.info("Attempting fallback to Gemini API after OpenRouter exhaustion...")
                try:
                    fallback_client = SupervisorLLMClient(provider="gemini", api_key=gemini_key)
                    return fallback_client.decide_next_experiment(prompt, experiment_id)
                except Exception as e:
                    logger.warning("Gemini fallback also failed: %s", e)
        elif self.provider == "gemini":
            or_key = get_openrouter_api_key()
            if or_key:
                logger.info("Attempting fallback to OpenRouter API after Gemini exhaustion...")
                try:
                    fallback_client = SupervisorLLMClient(provider="openrouter", api_key=or_key)
                    return fallback_client.decide_next_experiment(prompt, experiment_id)
                except Exception as e:
                    logger.warning("OpenRouter fallback also failed: %s", e)
        return None

    def _mock_decision(self, experiment_id: int) -> SupervisorDecision:
        """Deterministic, progression-based decision for offline testing and simulations."""
        if experiment_id <= 3:
            depth = 6
            l2 = 3.0 + (experiment_id * 1.0)
            lr = 0.05
            enable_interactions = False
            drop_features = []
            hyp = f"Run {experiment_id}: Capacity exploration with depth={depth} and initial L2 leaf reg={l2}."
            math_just = "Establishing benchmark and measuring initial generalization gap."
        elif experiment_id <= 7:
            depth = 4
            l2 = 6.0 + (experiment_id * 1.5)
            lr = 0.04
            enable_interactions = (experiment_id >= 5)
            drop_features = ["gender", "PhoneService"] if experiment_id >= 6 else []
            hyp = f"Run {experiment_id}: Reducing tree depth to {depth} and pruning low-signal features."
            math_just = "Overfitting index Omega > 15% indicates need for structural tree regularization."
        elif experiment_id <= 11:
            depth = 5
            l2 = 10.0
            lr = 0.025
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


class GeminiSupervisorClient(SupervisorLLMClient):
    """Backward-compatible alias for SupervisorLLMClient."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 3,
        base_retry_delay: float = 2.0,
    ):
        super().__init__(
            provider="auto",
            api_key=api_key,
            model_name=model_name,
            max_retries=max_retries,
            base_retry_delay=base_retry_delay,
        )
