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

    @staticmethod
    def compute_threshold_sweep(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Calculates confusion matrices, precision, recall, F1, and intervention rate across a range of thresholds."""
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        total = len(y_true)
        sweep = []
        for t in thresholds:
            pred = (y_prob >= t).astype(int)
            tp = int(np.sum((pred == 1) & (y_true == 1)))
            fp = int(np.sum((pred == 1) & (y_true == 0)))
            fn = int(np.sum((pred == 0) & (y_true == 1)))
            tn = int(np.sum((pred == 0) & (y_true == 0)))

            prec = float(tp / max(tp + fp, 1))
            rec = float(tp / max(tp + fn, 1))
            spec = float(tn / max(tn + fp, 1))
            f1 = float(2 * (prec * rec) / max(prec + rec, 1e-6))
            bal_acc = float((rec + spec) / 2.0)
            intervention_rate = float((tp + fp) / max(total, 1))
            acc = float((tp + tn) / max(total, 1))

            sweep.append({
                "threshold": round(float(t), 2),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "specificity": round(spec, 4),
                "f1_score": round(f1, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "churn_intervention_rate": round(intervention_rate, 4),
                "accuracy": round(acc, 4),
            })
        return sweep

    @staticmethod
    def compute_prediction_distribution_stats(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Calculates statistical moments and separation margin between churn and non-churn distributions."""
        mean_prob = float(np.mean(y_prob))
        std_prob = float(np.std(y_prob))

        churn_mask = (y_true == 1)
        non_churn_mask = (y_true == 0)

        churn_mean = float(np.mean(y_prob[churn_mask])) if np.sum(churn_mask) > 0 else 0.0
        non_churn_mean = float(np.mean(y_prob[non_churn_mask])) if np.sum(non_churn_mask) > 0 else 0.0
        separation_margin = churn_mean - non_churn_mean

        churn_median = float(np.median(y_prob[churn_mask])) if np.sum(churn_mask) > 0 else 0.0
        non_churn_median = float(np.median(y_prob[non_churn_mask])) if np.sum(non_churn_mask) > 0 else 0.0

        quantiles = [float(q) for q in np.quantile(y_prob, [0.1, 0.25, 0.5, 0.75, 0.9])]

        return {
            "overall_mean": round(mean_prob, 4),
            "overall_std": round(std_prob, 4),
            "churn_class_mean": round(churn_mean, 4),
            "non_churn_class_mean": round(non_churn_mean, 4),
            "churn_class_median": round(churn_median, 4),
            "non_churn_class_median": round(non_churn_median, 4),
            "separation_margin": round(separation_margin, 4),
            "quantiles_p10_p25_p50_p75_p90": [round(q, 4) for q in quantiles],
        }

    @staticmethod
    def compute_lift_and_gain_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
        """Calculates decile cumulative gains and lift analysis for targeting efficiency."""
        total_samples = len(y_true)
        total_churners = int(np.sum(y_true == 1))
        if total_churners == 0 or total_samples == 0:
            return []

        overall_churn_rate = total_churners / total_samples
        sorted_indices = np.argsort(y_prob)[::-1]
        sorted_true = y_true[sorted_indices]

        bin_size = total_samples / n_bins
        gain_points = []
        cumulative_churners = 0

        for b in range(1, n_bins + 1):
            end_idx = int(round(b * bin_size))
            start_idx = int(round((b - 1) * bin_size))
            bin_true = sorted_true[start_idx:end_idx]

            bin_churners = int(np.sum(bin_true == 1))
            cumulative_churners += bin_churners

            pct_contacted = b / n_bins
            cumulative_gain = cumulative_churners / total_churners
            bin_churn_rate = bin_churners / max(len(bin_true), 1)
            decile_lift = bin_churn_rate / max(overall_churn_rate, 1e-6)
            cumulative_lift = cumulative_gain / max(pct_contacted, 1e-6)

            gain_points.append({
                "decile": b,
                "percentage_contacted": round(pct_contacted * 100, 1),
                "decile_churners": bin_churners,
                "cumulative_churners": cumulative_churners,
                "cumulative_gain_pct": round(cumulative_gain * 100, 2),
                "decile_lift": round(decile_lift, 2),
                "cumulative_lift": round(cumulative_lift, 2),
            })
        return gain_points

    @staticmethod
    def compute_cohort_slice_metrics(
        df: Optional[pd.DataFrame],
        y_true: np.ndarray,
        y_prob: np.ndarray,
        slice_cols: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Calculates slice-level diagnostics across key customer segments."""
        if df is None:
            return {}

        if slice_cols is None:
            slice_cols = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "gender"]

        available_cols = [c for c in slice_cols if c in df.columns]
        if not available_cols:
            return {}

        slice_results = {}
        for col in available_cols:
            col_series = df[col].astype(str)
            categories = col_series.unique()
            cat_metrics = []

            for cat in categories:
                mask = (col_series == cat).to_numpy()
                count = int(np.sum(mask))
                if count < 5:
                    continue

                sub_true = y_true[mask]
                sub_prob = y_prob[mask]

                churn_count = int(np.sum(sub_true == 1))
                churn_rate = churn_count / count
                mean_p = float(np.mean(sub_prob))

                # Slice ROC-AUC if both classes present in slice
                slice_auc = None
                if len(np.unique(sub_true)) > 1:
                    try:
                        slice_auc = round(float(roc_auc_score(sub_true, sub_prob)), 4)
                    except Exception:
                        pass

                # False negative rate in slice at 0.5 threshold
                pred_05 = (sub_prob >= 0.5).astype(int)
                tp = np.sum((pred_05 == 1) & (sub_true == 1))
                fn = np.sum((pred_05 == 0) & (sub_true == 1))
                fnr = round(float(fn / max(fn + tp, 1)), 4)

                cat_metrics.append({
                    "segment": str(cat),
                    "sample_count": count,
                    "churn_count": churn_count,
                    "actual_churn_rate": round(churn_rate, 4),
                    "model_mean_prob": round(mean_p, 4),
                    "slice_roc_auc": slice_auc,
                    "slice_fnr_05": fnr,
                })

            cat_metrics.sort(key=lambda x: x["sample_count"], reverse=True)
            slice_results[col] = cat_metrics

        return slice_results

    @classmethod
    def compute_classification_diagnostics(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        X_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Calculates rich error analysis, calibration, distribution moments, and classification statistics."""
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

        # 5. Advanced Process Metrics: Threshold sweep, distribution moments, decile gains, and cohort slices
        threshold_sweep = cls.compute_threshold_sweep(y_true, y_prob)
        dist_stats = cls.compute_prediction_distribution_stats(y_true, y_prob)
        gain_curve = cls.compute_lift_and_gain_curve(y_true, y_prob)
        cohort_slices = cls.compute_cohort_slice_metrics(X_df, y_true, y_prob)

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
            "threshold_sweep": threshold_sweep,
            "distribution_stats": dist_stats,
            "separation_margin": dist_stats.get("separation_margin", 0.0),
            "lift_and_gain_curve": gain_curve,
            "cohort_slice_metrics": cohort_slices,
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
                "model_family": str(hyp.get("model_family", "catboost")),
                "oof_roc_auc": auc,
                "oof_pr_auc": float(metrics.get("oof_pr_auc", 0.0)),
                "oof_log_loss": float(metrics.get("oof_log_loss", 0.0)),
                "oof_brier_score": float(metrics.get("oof_brier_score", 0.0)),
                "overfitting_index": float(metrics.get("overfitting_index", 0.0)),
                "generalization_gap": float(metrics.get("generalization_gap", 0.0)),
                "ece_score": float(metrics.get("ece_score", 0.0)),
                "optimal_youden_threshold": float(metrics.get("optimal_youden_threshold", 0.5)),
                "separation_margin": float(metrics.get("separation_margin", 0.0)),
                "throughput_samples_per_sec": float(metrics.get("throughput_samples_per_sec", 0.0)),
                "inference_latency_ms": float(metrics.get("inference_latency_ms", 0.0)),
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

        delta_auc = champ_auc - base_auc
        delta_pr = champ_pr - base_pr
        delta_loss = champ_loss - base_loss
        delta_omega = champ_omega - base_omega

        curves_data = {
            "champion_experiment_id": champion_entry.get("experiment_id", 1),
            "champion_family": champion_entry.get("hyperparameters", {}).get("model_family", "catboost"),
            "roc_curve": champ_metrics.get("roc_curve_points", []),
            "pr_curve": champ_metrics.get("pr_curve_points", []),
            "calibration_curve": champ_metrics.get("calibration_curve_points", []),
            "probability_histogram": champ_metrics.get("probability_histogram", []),
            "threshold_sweep": champ_metrics.get("threshold_sweep", []),
            "lift_and_gain_curve": champ_metrics.get("lift_and_gain_curve", []),
            "distribution_stats": champ_metrics.get("distribution_stats", {}),
            "cohort_slice_metrics": champ_metrics.get("cohort_slice_metrics", {}),
        }

        summary_payload = {
            "total_experiments": len(completed_entries),
            "champion_experiment_id": champion_entry.get("experiment_id", 1),
            "champion_metrics": {
                "experiment_id": champion_entry.get("experiment_id", 1),
                "model_family": champion_entry.get("hyperparameters", {}).get("model_family", "catboost"),
                "oof_roc_auc": champ_auc,
                "oof_pr_auc": champ_pr,
                "oof_log_loss": champ_loss,
                "overfitting_index": champ_omega,
                "ece_score": float(champ_metrics.get("ece_score", 0.0)),
                "optimal_youden_threshold": float(champ_metrics.get("optimal_youden_threshold", 0.5)),
                "separation_margin": float(champ_metrics.get("separation_margin", 0.0)),
            },
            "baseline_metrics": {
                "experiment_id": baseline_entry.get("experiment_id", 1),
                "model_family": baseline_entry.get("hyperparameters", {}).get("model_family", "catboost"),
                "oof_roc_auc": base_auc,
                "oof_pr_auc": base_pr,
                "oof_log_loss": base_loss,
                "overfitting_index": base_omega,
            },
            "net_improvements": {
                "delta_roc_auc": round(delta_auc, 5),
                "pct_gain_roc_auc": round((delta_auc / max(base_auc, 1e-6)) * 100, 2),
                "delta_pr_auc": round(delta_pr, 5),
                "pct_gain_pr_auc": round((delta_pr / max(base_pr, 1e-6)) * 100, 2),
                "delta_log_loss": round(delta_loss, 4),
                "delta_overfitting_index": round(delta_omega, 2),
            },
            "trajectory_time_series": time_series,
            "curves_data": curves_data,
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        logger.info("Saved visualizable trajectory summary to %s", summary_path)
        return summary_path
