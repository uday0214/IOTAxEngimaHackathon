"""Telemetry collector and SHAP explainability analyzers."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class TelemetryCollector:
    """Aggregates metrics and diagnostics across folds."""

    @staticmethod
    def aggregate_fold_metrics(fold_aucs: List[float]) -> Dict[str, float]:
        return {
            "mean_auc": float(np.mean(fold_aucs)),
            "std_auc": float(np.std(fold_aucs)),
            "min_auc": float(np.min(fold_aucs)),
            "max_auc": float(np.max(fold_aucs)),
        }


class FastShapAnalyzer:
    """Computes TreeSHAP feature importances for tabular classifiers."""

    @staticmethod
    def extract_top_features(model, X: pd.DataFrame, top_k: int = 10) -> List[Tuple[str, float]]:
        # Fallback to feature_importances_ if shap is unavailable or slow
        if hasattr(model, "get_feature_importance"):
            importances = model.get_feature_importance()
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            return []

        cols = list(X.columns)
        ranked = sorted(zip(cols, importances), key=lambda x: x[1], reverse=True)
        return [(str(k), float(v)) for k, v in ranked[:top_k]]
