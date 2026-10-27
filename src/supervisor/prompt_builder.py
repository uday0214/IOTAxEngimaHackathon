"""System prompts and prompt construction engine for the Google AI Studio Gemini Supervisor."""

import json
from typing import Dict, List, Optional, Any
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
2. Learning Curve Trajectory & Early Stopping:
   - Check early stopping utilization (best_iteration / total_iterations). If best_iteration == total_iterations, the model saturated trees; increase iterations. If best_iteration << total_iterations / 4, learning rate is too high.
   - Check validation loss tail slope. If negative, training had not plateaued.
3. Feature Representation & Pruning (TreeSHAP vs Mutual Information):
   - Review top vs bottom features. Prune zero-importance or noisy features using feature_config.drop_low_importance_features.
4. Class Imbalance, Calibration & Thresholding:
   - Check Expected Calibration Error (ECE) and false negative rate (missed churners). If PR-AUC is lagging, switch class_weight_mode ('Balanced' or 'SqrtBalanced') or tune bagging_temperature.

=========================================
OUTPUT CONTRACT
=========================================
You must output a strictly valid JSON response adhering to the SupervisorDecision schema.
Do NOT enclose your output in markdown code blocks or explanatory conversational prose; output ONLY the JSON object.
Ensure your output strictly adheres to this structure:
{
  "experiment_id": <int>,
  "hypothesis": "<detailed scientific hypothesis citing telemetry>",
  "mathematical_justification": "<explicit quantitative reasoning citing Overfitting Index / Generalization Gap>",
  "hyperparameters": {
    "model_family": "catboost",
    "learning_rate": <float between 0.005 and 0.3>,
    "depth": <int between 3 and 10>,
    "l2_leaf_reg": <float between 0.1 and 50.0>,
    "iterations": <int between 300 and 3000>,
    "subsample": <float between 0.4 and 1.0>,
    "bagging_temperature": <float between 0.0 and 1.0>,
    "random_strength": <float between 0.0 and 10.0>,
    "class_weight_mode": "None" | "Balanced" | "SqrtBalanced",
    "early_stopping_rounds": <int between 10 and 100>,
    "random_seed": 42
  },
  "feature_config": {
    "enable_charge_ratios": true | false,
    "enable_tenure_cohorts": true | false,
    "enable_service_density": true | false,
    "enable_interaction_terms": true | false,
    "drop_low_importance_features": ["feature_name", ...]
  },
  "expected_outcome": "<anticipated impact on ROC-AUC and PR-AUC>"
}
"""

    def build_prompt(
        self,
        experiment_id: int,
        history: List[Dict[str, Any]],
        current_telemetry: Dict[str, Any],
        math_diagnosis: MathDiagnosis,
        best_auc: float,
    ) -> str:
        remaining_budget = 15 - experiment_id + 1

        # Format loss trajectory checkpoints
        trajectories = current_telemetry.get("loss_trajectory_samples", [])
        traj_str = "  " + "\n  ".join([
            f"Iter {p.get('iteration')}: Train Loss = {p.get('train_loss')}, Val Loss = {p.get('val_loss')}"
            for p in trajectories[:6]
        ]) if trajectories else "  (No checkpoints available)"

        # Format top & bottom features
        top_str = ", ".join([f"{f.get('feature')} ({f.get('importance')}%)" for f in current_telemetry.get("top_features", [])[:4]])
        bottom_str = ", ".join([f.get("feature") for f in current_telemetry.get("bottom_features", [])[:4]])

        prompt = f"""=== EXPERIMENT PLANNING CYCLE: PREPARING RUN {experiment_id} OF 15 ===
Remaining Experiment Budget: {remaining_budget} runs (STRICT LIMIT: Cannot exceed 15 runs).

[CURRENT BEST BENCHMARK]
Best 5-Fold OOF ROC-AUC achieved so far: {best_auc:.5f}

=======================================================
EXHAUSTIVE TRAINING PROCESS TELEMETRY (Run {experiment_id - 1})
=======================================================
1. Out-Of-Fold Predictive Metrics:
   - 5-Fold OOF ROC-AUC:      {current_telemetry.get('oof_roc_auc', 0.0):.5f}
   - 5-Fold OOF PR-AUC:       {current_telemetry.get('oof_pr_auc', 0.0):.5f}
   - Binary Log Loss:         {current_telemetry.get('oof_log_loss', 0.0):.4f}
   - Brier Calibration Score: {current_telemetry.get('oof_brier_score', 0.0):.4f}
   - Macro F1 (at 0.5):       {current_telemetry.get('f1_macro_05', 0.0):.4f}
   - F1 (at Youden J* thresh):{current_telemetry.get('f1_at_optimal', 0.0):.4f}
   - Matthews Corr Coef (MCC):{current_telemetry.get('matthews_corrcoef', 0.0):.4f}

2. Cross-Validation Stability:
   - Fold ROC-AUCs:           {current_telemetry.get('fold_roc_aucs', [])}
   - Fold Std Deviation:      {current_telemetry.get('fold_std_roc_auc', 0.0):.5f} ({math_diagnosis.fold_stability_assessment})

3. Learning Curve Dynamics & Early Stopping:
   - Final Train Loss:        {current_telemetry.get('final_train_loss', 0.0):.4f}
   - Final Val Loss:          {current_telemetry.get('final_val_loss', 0.0):.4f}
   - Generalization Gap:      {math_diagnosis.generalization_gap:.4f}
   - Overfitting Index (Omega):{math_diagnosis.overfitting_index:.2f}%
   - Early Stopping Iter:     {current_telemetry.get('best_iteration', 0)} / {current_telemetry.get('total_iterations', 1000)} ({math_diagnosis.early_stopping_fraction*100:.1f}% of budget)
   - Val Loss Tail Slope:     {math_diagnosis.val_loss_gradient_tail:.6f}
   - Loss Curve Trajectory Samples:
{traj_str}

4. Feature Signal Spectrum & Importances:
   - Dominant High-Signal Features: {top_str or 'N/A'}
   - Low-Signal Stagnant Features:  {bottom_str or 'N/A'}
   - Zero-Importance Features:      {current_telemetry.get('zero_importance_features', [])}

5. Calibration & Error Breakdown:
   - Expected Calibration Error:    {math_diagnosis.ece_score:.4f} ({math_diagnosis.calibration_status})
   - Optimal Youden Threshold J*:   {math_diagnosis.optimal_youden_threshold:.3f}
   - False Negative Rate (Missed):  {current_telemetry.get('false_negative_rate', 0.0)*100:.1f}%
   - False Positive Rate:           {current_telemetry.get('false_positive_rate', 0.0)*100:.1f}%

=======================================================
MATHEMATICAL INFERENCE REPORT & SYNTHESIS
=======================================================
- Paired Fold p-value vs Best Run: {math_diagnosis.delong_p_value_vs_best}
- Early Stopping Status:           {math_diagnosis.early_stopping_analysis}
- Strategic Engine Recommendation: {math_diagnosis.recommended_action}

=======================================================
TASK FOR EXPERIMENT {experiment_id}
=======================================================
1. Mathematically evaluate the Overfitting Index, early stopping utilization, and feature signals.
2. Formulate a hypothesis explaining what hyperparameters or feature toggles need to change to beat ROC-AUC {best_auc:.5f}.
3. Propose any zero/low-signal features to prune in `drop_low_importance_features`.
4. Output your decision strictly following the SupervisorDecision JSON schema.
"""
        return prompt
