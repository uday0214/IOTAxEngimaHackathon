"""Unit tests for AutonomousExperimentLoop and CatBoost model integration."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.orchestrator.loop import AutonomousExperimentLoop
from src.models.catboost_model import CatBoostPipelineModel
from src.features.pipeline import DynamicFeaturePipeline
from src.supervisor.schemas import SupervisorDecision, HyperparameterConfig, FeatureEngineeringConfig
from scripts.run_autonomous_loop import generate_sample_tabular_data


@pytest.fixture
def sample_data():
    df = generate_sample_tabular_data(n_samples=120)
    return df


@pytest.fixture
def temp_env(tmp_path):
    ledger_path = tmp_path / "experiment_ledger.jsonl"
    manifest_path = tmp_path / "dataset_manifest.json"
    canonical_path = tmp_path / "train_val.parquet"
    telemetry_dir = tmp_path / "telemetry"

    # Save small canonical parquet
    df = generate_sample_tabular_data(n_samples=100)
    df.to_parquet(canonical_path, index=False)

    # Save manifest
    from src.data.validator import DataProvenanceGuard
    manifest_data = {
        "raw_file_sha256": "dummy",
        "canonical_train_sha256": DataProvenanceGuard.compute_file_sha256(canonical_path),
        "content_row_hash": DataProvenanceGuard.compute_dataframe_content_hash(df),
        "total_rows": len(df),
        "allowed_indices": list(df.index),
        "allowed_columns": list(df.columns),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    return {
        "ledger_path": ledger_path,
        "manifest_path": manifest_path,
        "canonical_path": canonical_path,
        "telemetry_dir": telemetry_dir,
        "df": df,
    }


def test_orchestrator_initialization(temp_env):
    loop = AutonomousExperimentLoop(
        max_runs=5,
        ledger_path=temp_env["ledger_path"],
        manifest_path=temp_env["manifest_path"],
        canonical_path=temp_env["canonical_path"],
        telemetry_dir=temp_env["telemetry_dir"],
        execute_live_training=True,
        device="cpu",
    )
    # Verify CatBoost model class is loaded by default
    assert loop.model_cls == CatBoostPipelineModel
    assert loop.execute_live_training is True
    assert loop.device == "CPU"


def test_orchestrator_live_catboost_single_experiment(temp_env):
    loop = AutonomousExperimentLoop(
        max_runs=5,
        ledger_path=temp_env["ledger_path"],
        manifest_path=temp_env["manifest_path"],
        canonical_path=temp_env["canonical_path"],
        telemetry_dir=temp_env["telemetry_dir"],
        execute_live_training=True,
        device="cpu",
    )

    decision = SupervisorDecision(
        experiment_id=1,
        hypothesis="Test baseline with fast iterations",
        mathematical_justification="Verify live CatBoost integration in orchestrator",
        hyperparameters=HyperparameterConfig(
            model_family="catboost",
            iterations=300,
            depth=4,
            learning_rate=0.08,
            l2_leaf_reg=3.0,
        ),
        feature_config=FeatureEngineeringConfig(
            enable_charge_ratios=True,
            enable_tenure_cohorts=False,
            enable_service_density=False,
            enable_interaction_terms=False,
        ),
        expected_outcome="Telemetry captured and ledger recorded",
    )

    entry = loop.run_single_experiment(1, decision, temp_env["df"])

    assert entry["experiment_id"] == 1
    assert "metrics" in entry
    metrics = entry["metrics"]
    assert 0.0 <= metrics["oof_roc_auc"] <= 1.0
    assert 0.0 <= metrics["oof_pr_auc"] <= 1.0
    assert metrics["final_train_loss"] > 0
    assert metrics["final_val_loss"] > 0
    assert len(metrics["fold_metrics"]) == 5

    # Check ledger file was written
    assert temp_env["ledger_path"].exists()
    runs = loop.ledger.load_all_runs()
    assert len(runs) == 1
    assert runs[0]["experiment_id"] == 1


def test_orchestrator_champion_holdout_and_submission(temp_env, tmp_path):
    loop = AutonomousExperimentLoop(
        max_runs=5,
        ledger_path=temp_env["ledger_path"],
        manifest_path=temp_env["manifest_path"],
        canonical_path=temp_env["canonical_path"],
        telemetry_dir=temp_env["telemetry_dir"],
        execute_live_training=True,
        device="cpu",
    )

    decision = SupervisorDecision(
        experiment_id=1,
        hypothesis="Champion test fit",
        mathematical_justification="Fit champion on small df",
        hyperparameters=HyperparameterConfig(iterations=300, depth=4),
        feature_config=FeatureEngineeringConfig(),
        expected_outcome="Champion persisted",
    )
    loop.run_single_experiment(1, decision, temp_env["df"])

    # Fit champion with feature transformation matching decision
    champion_path = tmp_path / "champion_model.cbm"
    pipeline = DynamicFeaturePipeline(decision.feature_config)
    transformed_train = pipeline.transform(temp_env["df"])
    champion = loop.model_cls(config=decision.hyperparameters, device="cpu")
    X = transformed_train.drop(columns=["Churn"])
    y = transformed_train["Churn"]
    champion.fit_final(X, y)
    champion.save_model(champion_path)
    loop.champion_pipeline_model = champion

    # Test holdout evaluation
    holdout_report_path = tmp_path / "holdout_eval.json"
    holdout_df = generate_sample_tabular_data(n_samples=40)
    report = loop.evaluate_champion_holdout(holdout_df=holdout_df, output_path=holdout_report_path)
    assert holdout_report_path.exists()
    assert report["holdout_samples"] == 40
    assert 0.0 <= report["holdout_roc_auc"] <= 1.0

    # Test submission export
    sub_path = tmp_path / "submission.csv"
    sub_df = loop.export_submission(eval_df=holdout_df, output_path=sub_path)
    assert sub_path.exists()
    assert len(sub_df) == 40
    assert "Churn_Probability" in sub_df.columns
    assert "Churn_Prediction" in sub_df.columns
