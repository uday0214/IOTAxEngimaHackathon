"""Constructs structured prompts with mathematical diagnostics for the Gemini Supervisor."""

import json
from typing import Dict, List, Optional
from src.supervisor.math_engine import MathDiagnosis


class SupervisorPromptBuilder:
    """Builds the comprehensive context prompt for Google Gemini LLM."""

    SYSTEM_PROMPT = """You are an Autonomous Senior AI Data Scientist competing in the Utkrishta 2026 AI/ML Hackathon.
Your mission is to maximize the 5-Fold Stratified Cross-Validation ROC-AUC and PR-AUC on the Telco Customer Churn dataset.
CRITICAL CONSTRAINT: You are operating under a HARD LIMIT of 15 model-training experiments. Every fit cycle costs 1 experiment.
You must use empirical feedback, learning curve dynamics, and mathematical diagnostics from completed experiments to propose targeted, rational mutations to the model configuration.
Your chosen classifier family has been locked to CatBoost following Pre-Flight inductive bias analysis.
Output your decision strictly adhering to the JSON schema."""

    def build_prompt(
        self,
        experiment_id: int,
        history: List[Dict],
        current_telemetry: Dict,
        math_diagnosis: MathDiagnosis,
        best_auc: float,
    ) -> str:
        remaining_budget = 15 - experiment_id + 1

        prompt = f"""=== EXPERIMENT PLANNING CYCLE: RUN {experiment_id} of 15 ===
Remaining Experiment Budget: {remaining_budget} runs

[CURRENT BEST BENCHMARK]
Best 5-Fold OOF ROC-AUC achieved so far: {best_auc:.5f}

[PREVIOUS RUN TELEMETRY (Run {experiment_id - 1})]
- OOF ROC-AUC: {current_telemetry.get('oof_roc_auc', 0.0):.5f}
- OOF PR-AUC: {current_telemetry.get('oof_pr_auc', 0.0):.5f}
- Final Train Loss: {current_telemetry.get('final_train_loss', 0.0):.4f}
- Final Validation Loss: {current_telemetry.get('final_val_loss', 0.0):.4f}

[MATHEMATICAL INFERENCE REPORT]
- Generalization Gap (Val - Train Loss): {math_diagnosis.generalization_gap:.4f}
- Overfitting Index (Gap / Val Loss): {math_diagnosis.overfitting_index:.2f}%
- DeLong Test p-value vs Best Run: {math_diagnosis.delong_p_value_vs_best}
- Expected Calibration Error (ECE): {math_diagnosis.ece_score:.4f}
- Optimal Youden Threshold: {math_diagnosis.optimal_youden_threshold:.3f}
- Top Feature Importances (TreeSHAP): {math_diagnosis.top_shap_features}
- Stagnant / Low-Signal Features: {math_diagnosis.stagnant_features}
- Engine Recommendation: {math_diagnosis.recommended_action}

[HISTORICAL EXPERIMENTS SUMMARY]
{json.dumps(history[-3:], indent=2) if history else "No prior history (First experiment)."}

[TASK]
Carefully analyze the mathematical diagnosis and overfitting index.
Formulate a scientific hypothesis and propose the hyperparameter and feature engineering mutations for Experiment {experiment_id}.
"""
        return prompt
