"""LightGBM classifier wrapper for baseline benchmarking and challenger analysis."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

from src.models.base import BaseModel, ExperimentTelemetry
from src.supervisor.schemas import HyperparameterConfig


class LightGBMPipelineModel(BaseModel):
    """LightGBM classifier with categorical label encoding and cross-validation."""

    def __init__(self, config: HyperparameterConfig, experiment_id: int = 2):
        self.config = config
        self.experiment_id = experiment_id

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> ExperimentTelemetry:
        X = X.copy()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_cols:
            X[col] = X[col].astype("category")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof_probs = np.zeros(len(X))
        fold_aucs = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = LGBMClassifier(
                n_estimators=self.config.iterations,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.depth,
                reg_lambda=self.config.l2_leaf_reg,
                subsample=self.config.subsample or 0.8,
                random_state=random_state + fold,
                verbose=-1,
            )

            model.fit(X_tr, y_tr, eval_set=[(X_va, y_val)])
            val_probs = model.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = val_probs
            fold_aucs.append(float(roc_auc_score(y_val, val_probs)))

        oof_roc_auc = float(roc_auc_score(y, oof_probs))
        oof_pr_auc = float(average_precision_score(y, oof_probs))
        oof_logloss = float(log_loss(y, oof_probs))
        oof_brier = float(brier_score_loss(y, oof_probs))

        return ExperimentTelemetry(
            experiment_id=self.experiment_id,
            oof_roc_auc=oof_roc_auc,
            oof_pr_auc=oof_pr_auc,
            oof_log_loss=oof_logloss,
            oof_brier_score=oof_brier,
            fold_roc_aucs=fold_aucs,
            confusion_matrix=confusion_matrix(y, (oof_probs >= 0.5).astype(int)).tolist(),
        )
