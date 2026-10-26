"""Unit tests for dataset provenance validation and forced canonical restoration."""

import pytest
import json
from pathlib import Path
import pandas as pd
from src.data.validator import DataProvenanceGuard, DatasetManifest


def test_data_provenance_verification(tmp_path: Path):
    manifest_file = tmp_path / "manifest.json"
    canonical_parquet = tmp_path / "train_val.parquet"

    # Create dummy canonical dataset
    df_canonical = pd.DataFrame({
        "tenure": [12, 24, 36],
        "MonthlyCharges": [50.0, 70.0, 90.0],
        "Churn": [0, 1, 0],
    })
    df_canonical.to_parquet(canonical_parquet)

    content_hash = DataProvenanceGuard.compute_dataframe_content_hash(df_canonical)
    file_hash = DataProvenanceGuard.compute_file_sha256(canonical_parquet)

    manifest_data = {
        "raw_file_sha256": "fake_raw_hash",
        "canonical_train_sha256": file_hash,
        "content_row_hash": content_hash,
        "total_rows": len(df_canonical),
        "allowed_indices": list(df_canonical.index),
        "allowed_columns": list(df_canonical.columns),
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f)

    guard = DataProvenanceGuard(manifest_path=manifest_file, canonical_data_path=canonical_parquet)

    # 1. Valid DataFrame passes
    assert guard.verify_integrity(df_canonical) is True

    # 2. Tampered DataFrame with extra/foreign row fails and gets restored
    tampered_df = pd.DataFrame({
        "tenure": [12, 24, 36, 99],
        "MonthlyCharges": [50.0, 70.0, 90.0, 150.0],
        "Churn": [0, 1, 0, 1],
    })
    assert guard.verify_integrity(tampered_df) is False

    restored_df = guard.assert_strictly_canonical(tampered_df)
    assert len(restored_df) == 3
    assert guard.verify_integrity(restored_df) is True
