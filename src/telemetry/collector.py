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
)

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
