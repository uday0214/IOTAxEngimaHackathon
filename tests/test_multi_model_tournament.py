"""Unit tests for Multi-Model Tournament and diverse model pipeline families."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models import (
    BaseModel,
    CatBoostPipelineModel,
    LightGBMPipelineModel,
    XGBoostPipelineModel,
    RandomForestPipelineModel,
    LogisticRegressionPipelineModel,
    get_model_class,
    create_model,
    load_champion_model,
)
from src.supervisor.schemas import HyperparameterConfig, SupervisorDecision, FeatureEngineeringConfig
from src.orchestrator.loop import AutonomousExperimentLoop, get_tournament_baseline_decision
from scripts.run_autonomous_loop import generate_sample_tabular_data


@pytest.fixture
def sample_dataset():
    """Generates small synthetic tabular customer churn dataset."""
    return generate_sample_tabular_data(n_samples=120)


@pytest.fixture
def temp_orchestrator_env(tmp_path):
    ledger_path = tmp_path / "experiment_ledger.jsonl"
    manifest_path = tmp_path / "dataset_manifest.json"
    canonical_path = tmp_path / "train_val.parquet"
    telemetry_dir = tmp_path / "telemetry"

    df = generate_sample_tabular_data(n_samples=100)
    df.to_parquet(canonical_path, index=False)

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


def test_model_factory_registry():
    """Verifies all required model families are retrievable from the factory."""
    families = ["catboost", "lightgbm", "xgboost", "random_forest", "logistic_regression"]
    for fam in families:
        cls = get_model_class(fam)
        assert issubclass(cls, BaseModel)

    with pytest.raises(ValueError):
        get_model_class("unknown_model_family")


def test_lightgbm_pipeline_cv_and_persistence(sample_dataset, tmp_path):
    """Tests LightGBM 5-fold CV, fit_final, predict_proba, and model save/load."""
    X = sample_dataset.drop(columns=["Churn"])
    y = sample_dataset["Churn"]

    cfg = HyperparameterConfig(
        model_family="lightgbm",
        iterations=50,
        depth=4,
        learning_rate=0.1,
        early_stopping_rounds=15,
    )
    model = LightGBMPipelineModel(config=cfg, experiment_id=1, device="cpu")

    telemetry = model.cross_validate(X, y, n_splits=3)
    assert 0.0 <= telemetry.oof_roc_auc <= 1.0
    assert 0.0 <= telemetry.oof_pr_auc <= 1.0
    assert len(telemetry.fold_metrics) == 3
    assert len(telemetry.top_features) > 0

    # Test fit_final and persistence
    model.fit_final(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == len(X)
    assert 0.0 <= probs.min() and probs.max() <= 1.0

    save_path = tmp_path / "lgbm_champion.joblib"
    model.save_model(save_path)
    assert save_path.exists()

    loaded = load_champion_model(save_path, model_family="lightgbm")
    loaded_probs = loaded.predict_proba(X)
    np.testing.assert_allclose(probs, loaded_probs, rtol=1e-5)


def test_xgboost_pipeline_cv_and_persistence(sample_dataset, tmp_path):
    """Tests XGBoost 5-fold CV, fit_final, predict_proba, and model save/load."""
    X = sample_dataset.drop(columns=["Churn"])
    y = sample_dataset["Churn"]

    cfg = HyperparameterConfig(
        model_family="xgboost",
        iterations=50,
        depth=4,
        learning_rate=0.1,
        early_stopping_rounds=15,
    )
    model = XGBoostPipelineModel(config=cfg, experiment_id=1, device="cpu")

    telemetry = model.cross_validate(X, y, n_splits=3)
    assert 0.0 <= telemetry.oof_roc_auc <= 1.0
    assert 0.0 <= telemetry.oof_pr_auc <= 1.0
    assert len(telemetry.fold_metrics) == 3

    model.fit_final(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == len(X)

    save_path = tmp_path / "xgb_champion.joblib"
    model.save_model(save_path)
    assert save_path.exists()

    loaded = load_champion_model(save_path, model_family="xgboost")
    loaded_probs = loaded.predict_proba(X)
    np.testing.assert_allclose(probs, loaded_probs, rtol=1e-5)


def test_random_forest_pipeline_cv_and_persistence(sample_dataset, tmp_path):
    """Tests RandomForest 5-fold CV with automated encoding pipeline and persistence."""
    X = sample_dataset.drop(columns=["Churn"])
    y = sample_dataset["Churn"]

    cfg = HyperparameterConfig(
        model_family="random_forest",
        iterations=50,
        depth=4,
        min_samples_split=2,
    )
    model = RandomForestPipelineModel(config=cfg, experiment_id=1, device="cpu")

    telemetry = model.cross_validate(X, y, n_splits=3)
    assert 0.0 <= telemetry.oof_roc_auc <= 1.0
    assert 0.0 <= telemetry.oof_pr_auc <= 1.0
    assert len(telemetry.fold_metrics) == 3

    model.fit_final(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == len(X)

    save_path = tmp_path / "rf_champion.joblib"
    model.save_model(save_path)
    assert save_path.exists()

    loaded = load_champion_model(save_path, model_family="random_forest")
    loaded_probs = loaded.predict_proba(X)
    np.testing.assert_allclose(probs, loaded_probs, rtol=1e-5)


def test_logistic_regression_pipeline_cv_and_persistence(sample_dataset, tmp_path):
    """Tests Logistic Regression pipeline with scaling/encoding and persistence."""
    X = sample_dataset.drop(columns=["Churn"])
    y = sample_dataset["Churn"]

    cfg = HyperparameterConfig(
        model_family="logistic_regression",
        C=1.0,
        penalty="l2",
        solver="lbfgs",
    )
    model = LogisticRegressionPipelineModel(config=cfg, experiment_id=1, device="cpu")

    telemetry = model.cross_validate(X, y, n_splits=3)
    assert 0.0 <= telemetry.oof_roc_auc <= 1.0
    assert 0.0 <= telemetry.oof_pr_auc <= 1.0
    assert len(telemetry.fold_metrics) == 3

    model.fit_final(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == len(X)

    save_path = tmp_path / "lr_champion.joblib"
    model.save_model(save_path)
    assert save_path.exists()

    loaded = load_champion_model(save_path, model_family="logistic_regression")
    loaded_probs = loaded.predict_proba(X)
    np.testing.assert_allclose(probs, loaded_probs, rtol=1e-5)


def test_tournament_baseline_decisions():
    """Verifies baseline decision generation for all 5 tournament families."""
    families = ["catboost", "lightgbm", "xgboost", "random_forest", "logistic_regression"]
    for i, fam in enumerate(families, start=1):
        decision = get_tournament_baseline_decision(i, fam)
        assert decision.experiment_id == i
        assert decision.hyperparameters.model_family == fam
        assert len(decision.hypothesis) > 0


def test_tournament_orchestrator_simulation_loop(temp_orchestrator_env, monkeypatch):
    """
    Verifies that the autonomous loop:
    1. Runs one experiment for each candidate model family during tournament (Runs 1-5).
    2. Crowns the top-performing family as tournament champion.
    3. Locks that winning architecture for all remaining experiments (Runs 6-15).
    4. Adheres strictly to the 15-experiment hard budget limit.
    """
    monkeypatch.setenv("MOCK_SUPERVISOR", "true")
    candidates = ["catboost", "lightgbm", "xgboost", "random_forest", "logistic_regression"]
    loop = AutonomousExperimentLoop(
        max_runs=15,
        ledger_path=temp_orchestrator_env["ledger_path"],
        manifest_path=temp_orchestrator_env["manifest_path"],
        canonical_path=temp_orchestrator_env["canonical_path"],
        telemetry_dir=temp_orchestrator_env["telemetry_dir"],
        execute_live_training=False,  # Simulation mode for fast testing
        tournament_candidates=candidates,
    )

    results = loop.execute_full_15_run_loop(train_val_df=temp_orchestrator_env["df"])

    assert results["total_runs"] == 15
    runs = loop.ledger.load_all_runs()
    assert len(runs) == 15

    # Check tournament phase: First 5 runs tested each candidate model family
    tournament_families = [r["hyperparameters"]["model_family"] for r in runs[:5]]
    assert tournament_families == candidates

    # Check tournament leaderboard was created and winner selected
    assert len(loop.tournament_leaderboard) == 5
    assert loop.winning_family is not None
    assert loop.winning_family in candidates

    # Check that subsequent runs (6 to 15) were locked to the winning family
    for r in runs[5:]:
        assert r["hyperparameters"]["model_family"] == loop.winning_family

    # Verify report JSON contains tournament leaderboard
    report_json_path = temp_orchestrator_env["telemetry_dir"].parent / "supervisor_final_report.json"
    assert report_json_path.exists()
    with open(report_json_path, encoding="utf-8") as f:
        report_data = json.load(f)
    assert "tournament_leaderboard" in report_data
    assert report_data["winning_family"] == loop.winning_family
