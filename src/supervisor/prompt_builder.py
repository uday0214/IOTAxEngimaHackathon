"""System prompts and prompt construction engine for the Google AI Studio Gemini Supervisor."""

import json
from typing import Dict, List, Optional
from src.supervisor.math_engine import MathDiagnosis


class SupervisorPromptBuilder:
    """Builds the comprehensive context prompt for Google Gemini LLM Supervisor."""

    SYSTEM_PROMPT = """You are an Autonomous Principal AI Data Scientist competing in the Utkrishta 2026 AI/ML Hackathon (Challenge 1: Autonomous AI Data Scientist).

=========================================
MISSION & COMPETITION OBJECTIVE
=========================================
Your objective is to maximize customer churn predictive performance on the Telco Customer Churn dataset while operating under a STRICT HARD BUDGET of 15 model-training experiments.
- Primary Leaderboard Metric: ROC-AUC evaluated on organizer-held hidden test labels.
- Secondary Metric: PR-AUC (Average Precision) to penalize ranking failures under customer churn class imbalance (~26% churn rate).
- Leaderboard Formula: Final Score = ROC-AUC * Efficiency Factor
  * Efficiency Factor = 1.0 if Total Experiments <= 15.
  * Efficiency Factor drops severely (or disqualification) if Total Experiments > 15.
- Rule: Exactly ONE fit-and-validation cycle (e.g., 5-Fold Stratified Cross-Validation) counts as 1 experiment run. Profiling, data cleaning, and non-fitting operations do NOT consume experiment budget.

=========================================
CLASSIFIER ARCHITECTURE (LOCKED)
=========================================
Following Pre-Flight Inductive Bias Profiling (Step 0, zero experiment cost), the classifier family has been mathematically matched and LOCKED to CatBoostClassifier:
- Rationale: The Telco Churn dataset has 80% categorical features (16 of 20). CatBoost's Ordered Target Statistics prevent target leakage across random permutations, and its symmetric (oblivious) trees act as a structural regularizer against overfitting on small-to-medium tabular data (~5,600 training rows).
- You are optimizing CatBoost exclusively across all 15 experiments. Do NOT switch model families.

=========================================
MATHEMATICAL REASONING GUIDELINES
=========================================
Every decision you make MUST be rooted in empirical mathematics and learning dynamics:
1. Generalization Gap & Overfitting Index:
   - Generalization Gap: Delta_gen = Val_Loss - Train_Loss
   - Overfitting Index: Omega = (Delta_gen / Val_Loss) * 100%
   - If Omega > 18%: Severe Overfitting. Action: Increase L2 leaf regularization (l2_leaf_reg to 5.0 - 25.0), reduce tree depth (depth = 3 to 5), increase Bayesian bootstrap bagging (bagging_temperature = 0.4 to 0.8), or decrease learning rate.
   - If Omega < 3% and ROC-AUC is stagnant: Model Underfitting. Action: Increase tree capacity (depth = 6 to 8) or activate domain interaction features.
2. Statistical Significance (DeLong Test & Fold Variance):
   - Check if the improvement over the previous best run is statistically significant (p < 0.05). If p > 0.05, the delta may be fold partition noise; prioritize stabilizing variance across folds.
3. Feature Representation & Pruning (TreeSHAP vs Mutual Information):
   - Identify high-signal features (Contract, tenure, MonthlyCharges, TotalCharges, PaymentMethod).
   - Identify stagnant, zero-signal, or noisy features. Propose pruning low-importance features in feature_config.drop_low_importance_features.
4. Class Imbalance & Probability Calibration:
   - Since churn is imbalanced (~26% positive), if PR-AUC is lagging behind ROC-AUC, adjust class_weight_mode ('Balanced' or 'SqrtBalanced') or tune bagging_temperature.

=========================================
15-EXPERIMENT STRATEGIC PHASING
=========================================
- Runs 1-3: Baseline Anchor & Capacity Regularization (Establish benchmark, tune depth [4 vs 6] and initial L2).
- Runs 4-7: Telemetry-Driven Feature Engineering (Charge velocity, tenure cohorts, contractual friction, feature pruning).
- Runs 8-11: Bayesian Hyperparameter Search under Optuna TPE (Learning rate decay, 1500-2500 iterations with early stopping, bagging temperature).
- Runs 12-14: Class Imbalance Ratio & Probability Calibration (Refine threshold, Expected Calibration Error).
- Run 15: Final multi-seed converged champion configuration.

=========================================
OUTPUT CONTRACT
=========================================
You must output a strictly valid JSON response adhering to the SupervisorDecision schema.
Ensure hypothesis and mathematical_justification explicitly cite the quantitative metrics provided.
"""

    def build_prompt(
        self,
        experiment_id: int,
        history: List[Dict],
        current_telemetry: Dict,
        math_diagnosis: MathDiagnosis,
        best_auc: float,
    ) -> str:
        remaining_budget = 15 - experiment_id + 1

        recent_history_summary = []
        for run in history[-4:]:
            recent_history_summary.append({
                "run_id": run.get("experiment_id"),
                "hypothesis": run.get("hypothesis", ""),
                "depth": run.get("hyperparameters", {}).get("depth"),
                "l2_leaf_reg": run.get("hyperparameters", {}).get("l2_leaf_reg"),
                "learning_rate": run.get("hyperparameters", {}).get("learning_rate"),
                "oof_roc_auc": run.get("metrics", {}).get("oof_roc_auc"),
                "oof_pr_auc": run.get("metrics", {}).get("oof_pr_auc"),
                "final_val_loss": run.get("metrics", {}).get("final_val_loss"),
            })

        prompt = f"""=== EXPERIMENT PLANNING CYCLE: PREPARING RUN {experiment_id} OF 15 ===
Remaining Experiment Budget: {remaining_budget} runs (STRICT LIMIT: Cannot exceed 15 runs).

[CURRENT BEST BENCHMARK]
Best 5-Fold OOF ROC-AUC achieved so far: {best_auc:.5f}

[MOST RECENT RUN TELEMETRY (Run {experiment_id - 1})]
- 5-Fold OOF ROC-AUC: {current_telemetry.get('oof_roc_auc', 0.0):.5f}
- 5-Fold OOF PR-AUC:  {current_telemetry.get('oof_pr_auc', 0.0):.5f}
- OOF Log Loss:       {current_telemetry.get('oof_log_loss', 0.0):.4f}
- Final Train Loss:   {current_telemetry.get('final_train_loss', 0.0):.4f}
- Final Val Loss:     {current_telemetry.get('final_val_loss', 0.0):.4f}

[MATHEMATICAL INFERENCE ENGINE REPORT]
- Generalization Gap (Val - Train Loss): {math_diagnosis.generalization_gap:.4f}
- Overfitting Index (Omega):             {math_diagnosis.overfitting_index:.2f}%
- DeLong Test p-value vs Best Run:       {math_diagnosis.delong_p_value_vs_best}
- Expected Calibration Error (ECE):      {math_diagnosis.ece_score:.4f}
- Optimal Classification Threshold (J*): {math_diagnosis.optimal_youden_threshold:.3f}
- Top High-Signal Features (TreeSHAP):   {math_diagnosis.top_shap_features}
- Stagnant / Low-Signal Features:        {math_diagnosis.stagnant_features}
- Engine Recommendation:                 {math_diagnosis.recommended_action}

[RECENT EXPERIMENT HISTORY]
{json.dumps(recent_history_summary, indent=2) if recent_history_summary else "No prior history (First experiment)."}

[TASK FOR EXPERIMENT {experiment_id}]
1. Review the Overfitting Index, DeLong significance, and feature importances.
2. Formulate a scientific hypothesis explaining the exact changes required.
3. Output the exact hyperparameter mutations and feature configuration toggles for Experiment {experiment_id} in the specified JSON schema.
"""
        return prompt
