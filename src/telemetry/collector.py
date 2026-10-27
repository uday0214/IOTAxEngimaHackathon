"""Extensive Telemetry Collector for Model Training & Learning Process Diagnostics."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


class ExtensiveTelemetryCollector:
    """Collects, aggregates, and persists extensive training process telemetry."""

    @staticmethod
    def compute_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
        """Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
        bin_limits = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        n = len(y_true)

        for i in range(n_bins):
            mask = (y_prob >= bin_limits[i]) & (y_prob < bin_limits[i + 1])
            if np.sum(mask) > 0:
                bin_acc = float(np.mean(y_true[mask]))
                bin_conf = float(np.mean(y_prob[mask]))
                diff = abs(bin_acc - bin_conf)
                ece += (np.sum(mask) / n) * diff
                mce = max(mce, diff)

        return float(ece), float(mce)

    @staticmethod
    def compute_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Finds classification threshold maximizing Youden's J statistic (Sensitivity + Specificity - 1)."""
        thresholds = np.linspace(0.1, 0.9, 81)
        best_j = -1.0
        best_t = 0.5

        for t in thresholds:
            pred = (y_prob >= t).astype(int)
            tp = np.sum((pred == 1) & (y_true == 1))
            tn = np.sum((pred == 0) & (y_true == 0))
            fn = np.sum((pred == 0) & (y_true == 1))
            fp = np.sum((pred == 1) & (y_true == 0))
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            j = sens + spec - 1.0
            if j > best_j:
                best_j = j
                best_t = float(t)

        return best_t

    @classmethod
    def compute_classification_diagnostics(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculates rich error analysis, calibration, and classification statistics."""
        oof_roc_auc = float(roc_auc_score(y_true, y_prob))
        oof_pr_auc = float(average_precision_score(y_true, y_prob))
        oof_logloss = float(log_loss(y_true, y_prob))
        oof_brier = float(brier_score_loss(y_true, y_prob))

        ece, mce = cls.compute_calibration_error(y_true, y_prob)
        opt_thresh = cls.compute_optimal_threshold(y_true, y_prob)

        # Metrics at 0.5
        pred_05 = (y_prob >= 0.5).astype(int)
        cm_05 = confusion_matrix(y_true, pred_05).tolist()
        f1_05 = float(f1_score(y_true, pred_05, average="macro"))

        # Metrics at optimal threshold
        pred_opt = (y_prob >= opt_thresh).astype(int)
        cm_opt = confusion_matrix(y_true, pred_opt).tolist()
        f1_opt = float(f1_score(y_true, pred_opt, average="macro"))
        mcc = float(matthews_corrcoef(y_true, pred_opt))

        # Rates at 0.5
        tn, fp, fn, tp = confusion_matrix(y_true, pred_05).ravel()
        fnr = float(fn / max(fn + tp, 1))  # Churners missed
        fpr = float(fp / max(fp + tn, 1))  # False alarms

        # 1. ROC curve coordinates (downsampled for plotting)
        roc_pts: List[Dict[str, float]] = []
        try:
            fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true, y_prob)
            if len(fpr_arr) > 25:
                indices = np.linspace(0, len(fpr_arr) - 1, 25, dtype=int)
                roc_pts = [
                    {
                        "fpr": round(float(fpr_arr[i]), 4),
                        "tpr": round(float(tpr_arr[i]), 4),
                        "threshold": round(float(thresh_arr[i]), 4),
                    }
                    for i in indices
                ]
            else:
                roc_pts = [
                    {"fpr": round(float(f), 4), "tpr": round(float(t), 4), "threshold": round(float(th), 4)}
                    for f, t, th in zip(fpr_arr, tpr_arr, thresh_arr)
                ]
        except Exception:
            pass

        # 2. Precision-Recall curve coordinates (downsampled for plotting)
        pr_pts: List[Dict[str, float]] = []
        try:
            prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
            if len(prec_arr) > 25:
                indices = np.linspace(0, len(prec_arr) - 1, 25, dtype=int)
                pr_pts = [
                    {"precision": round(float(prec_arr[i]), 4), "recall": round(float(rec_arr[i]), 4)}
                    for i in indices
                ]
            else:
                pr_pts = [
                    {"precision": round(float(p), 4), "recall": round(float(r), 4)}
                    for p, r in zip(prec_arr, rec_arr)
                ]
        except Exception:
            pass

        # 3. Probability calibration curve coordinates
        calib_pts: List[Dict[str, float]] = []
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            calib_pts = [
                {"prob_pred": round(float(p), 4), "prob_true": round(float(t), 4)}
                for p, t in zip(prob_pred, prob_true)
            ]
        except Exception:
            pass

        # 4. Prediction probability confidence distribution
        prob_hist: List[Dict[str, Any]] = []
        try:
            counts, bin_edges = np.histogram(y_prob, bins=10, range=(0.0, 1.0))
            prob_hist = [
                {
                    "bin_lower": round(float(bin_edges[i]), 2),
                    "bin_upper": round(float(bin_edges[i + 1]), 2),
                    "count": int(counts[i]),
                }
                for i in range(10)
            ]
        except Exception:
            pass

        return {
            "oof_roc_auc": oof_roc_auc,
            "oof_pr_auc": oof_pr_auc,
            "oof_log_loss": oof_logloss,
            "oof_brier_score": oof_brier,
            "ece_score": ece,
            "mce_score": mce,
            "optimal_youden_threshold": opt_thresh,
            "f1_macro_05": f1_05,
            "f1_at_optimal": f1_opt,
            "matthews_corrcoef": mcc,
            "confusion_matrix_05": cm_05,
            "confusion_matrix_optimal": cm_opt,
            "false_negative_rate": fnr,
            "false_positive_rate": fpr,
            "roc_curve_points": roc_pts,
            "pr_curve_points": pr_pts,
            "calibration_curve_points": calib_pts,
            "probability_histogram": prob_hist,
        }

    @staticmethod
    def process_feature_importance_spectrum(
        importances: Dict[str, float],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        """Sorts features by importance and identifies top, bottom, and zero-signal features."""
        sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_feats = [{"feature": k, "importance": round(v, 4)} for k, v in sorted_feats[:5]]
        bottom_feats = [{"feature": k, "importance": round(v, 4)} for k, v in sorted_feats[-5:]]
        zero_feats = [k for k, v in sorted_feats if v <= 1e-6]
        return top_feats, bottom_feats, zero_feats

    @staticmethod
    def sample_loss_trajectories(
        train_losses: List[float],
        val_losses: List[float],
        n_samples: int = 15,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Downsamples full learning loss trajectories and calculates tail validation gradient."""
        total_steps = len(val_losses)
        if total_steps == 0:
            return [], 0.0

        step_indices = np.linspace(0, total_steps - 1, min(n_samples, total_steps), dtype=int)
        sampled = []
        for idx in step_indices:
            sampled.append({
                "iteration": int(idx),
                "train_loss": round(float(train_losses[idx]), 4) if idx < len(train_losses) else None,
                "val_loss": round(float(val_losses[idx]), 4),
            })

        # Calculate tail validation slope over last 15% of iterations
        tail_len = max(5, int(total_steps * 0.15))
        tail_vals = val_losses[-tail_len:]
        x_tail = np.arange(len(tail_vals))
        tail_slope = float(np.polyfit(x_tail, tail_vals, 1)[0]) if len(tail_vals) > 1 else 0.0

        return sampled, tail_slope

    @staticmethod
    def save_detailed_telemetry(
        experiment_id: int,
        telemetry_dict: Dict[str, Any],
        output_dir: Path = Path("artifacts/telemetry"),
    ) -> Path:
        """Persists comprehensive telemetry JSON document to artifacts/telemetry/."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"exp_{experiment_id:02d}_telemetry.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(telemetry_dict, f, indent=2)
        logger.info("Saved extensive training telemetry artifact to %s", file_path)
        return file_path

    @staticmethod
    def save_trajectory_summary(
        completed_entries: List[Dict[str, Any]],
        output_dir: Path = Path("artifacts/telemetry"),
    ) -> Path:
        """Compiles and persists a full visualizable trajectory dataset across all experiments."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "full_trajectory_summary.json"

        if not completed_entries:
            return summary_path

        time_series = []
        best_auc = -1.0
        champion_entry = completed_entries[0]

        for entry in completed_entries:
            exp_id = entry.get("experiment_id", 0)
            metrics = entry.get("metrics", {})
            hyp = entry.get("hyperparameters", {})
            feat = entry.get("feature_config", {})

            auc = float(metrics.get("oof_roc_auc", 0.0))
            if auc > best_auc:
                best_auc = auc
                champion_entry = entry

            time_series.append({
                "experiment_id": exp_id,
                "oof_roc_auc": auc,
                "oof_pr_auc": float(metrics.get("oof_pr_auc", 0.0)),
                "oof_log_loss": float(metrics.get("oof_log_loss", 0.0)),
                "oof_brier_score": float(metrics.get("oof_brier_score", 0.0)),
                "overfitting_index": float(metrics.get("overfitting_index", 0.0)),
                "generalization_gap": float(metrics.get("generalization_gap", 0.0)),
                "ece_score": float(metrics.get("ece_score", 0.0)),
                "optimal_youden_threshold": float(metrics.get("optimal_youden_threshold", 0.5)),
                "depth": int(hyp.get("depth", 6)),
                "learning_rate": float(hyp.get("learning_rate", 0.05)),
                "l2_leaf_reg": float(hyp.get("l2_leaf_reg", 3.0)),
                "iterations": int(hyp.get("iterations", 1000)),
                "bagging_temperature": float(hyp.get("bagging_temperature", 0.2) or 0.2),
                "class_weight_mode": str(hyp.get("class_weight_mode", "None")),
                "enable_interaction_terms": bool(feat.get("enable_interaction_terms", False)),
                "pruned_features_count": len(feat.get("drop_low_importance_features", [])),
                "training_duration_seconds": float(metrics.get("training_duration_seconds", 0.0)),
            })

        baseline_entry = completed_entries[0]
        base_metrics = baseline_entry.get("metrics", {})
        champ_metrics = champion_entry.get("metrics", {})

        base_auc = float(base_metrics.get("oof_roc_auc", 0.0))
        champ_auc = float(champ_metrics.get("oof_roc_auc", 0.0))
        base_pr = float(base_metrics.get("oof_pr_auc", 0.0))
        champ_pr = float(champ_metrics.get("oof_pr_auc", 0.0))
        base_loss = float(base_metrics.get("oof_log_loss", 0.0))
        champ_loss = float(champ_metrics.get("oof_log_loss", 0.0))
        base_omega = float(base_metrics.get("overfitting_index", 0.0))
        champ_omega = float(champ_metrics.get("overfitting_index", 0.0))

        # Parameter correlations / sensitivities with ROC-AUC
        sensitivities = {}
        for param in ["depth", "learning_rate", "l2_leaf_reg", "iterations", "bagging_temperature"]:
            vals = [ts[param] for ts in time_series]
            aucs = [ts["oof_roc_auc"] for ts in time_series]
            if len(set(vals)) > 1 and len(vals) >= 3:
                try:
                    corr = float(np.corrcoef(vals, aucs)[0, 1])
                    sensitivities[param] = {
                        "correlation_with_auc": round(corr, 3) if not np.isnan(corr) else 0.0,
                        "unique_values_tested": sorted(list(set(vals))),
                    }
                except Exception:
                    pass

        summary_payload = {
            "total_experiments": len(completed_entries),
            "champion_experiment_id": champion_entry.get("experiment_id"),
            "baseline_metrics": {
                "oof_roc_auc": base_auc,
                "oof_pr_auc": base_pr,
                "oof_log_loss": base_loss,
                "overfitting_index": base_omega,
                "ece_score": float(base_metrics.get("ece_score", 0.0)),
            },
            "champion_metrics": {
                "oof_roc_auc": champ_auc,
                "oof_pr_auc": champ_pr,
                "oof_log_loss": champ_loss,
                "overfitting_index": champ_omega,
                "ece_score": float(champ_metrics.get("ece_score", 0.0)),
                "optimal_youden_threshold": float(champ_metrics.get("optimal_youden_threshold", 0.5)),
            },
            "net_improvements": {
                "delta_roc_auc": round(champ_auc - base_auc, 5),
                "pct_gain_roc_auc": round(((champ_auc - base_auc) / max(base_auc, 1e-6)) * 100.0, 2),
                "delta_pr_auc": round(champ_pr - base_pr, 5),
                "pct_gain_pr_auc": round(((champ_pr - base_pr) / max(base_pr, 1e-6)) * 100.0, 2),
                "delta_log_loss": round(champ_loss - base_loss, 4),
                "delta_overfitting_index": round(champ_omega - base_omega, 2),
            },
            "parameter_sensitivities": sensitivities,
            "trajectory_time_series": time_series,
            "curves_data": {
                "baseline_roc_curve": base_metrics.get("roc_curve_points", []),
                "champion_roc_curve": champ_metrics.get("roc_curve_points", []),
                "baseline_pr_curve": base_metrics.get("pr_curve_points", []),
                "champion_pr_curve": champ_metrics.get("pr_curve_points", []),
                "baseline_calibration_curve": base_metrics.get("calibration_curve_points", []),
                "champion_calibration_curve": champ_metrics.get("calibration_curve_points", []),
                "champion_probability_histogram": champ_metrics.get("probability_histogram", []),
                "champion_feature_importances": champ_metrics.get("feature_importances", {}),
            },
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        logger.info("Saved visualizable trajectory summary to %s", summary_path)
        return summary_path
