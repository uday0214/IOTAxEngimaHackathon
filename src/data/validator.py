"""Data Provenance & Cryptographic Validation Guard.

Verifies SHA-256 digests, row-wise content hash, cardinality, and set-theoretic index containment.
Forces restoration of the canonical partition if tampering or unauthorized data is detected.
"""

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class UnauthorizedDatasetException(RuntimeError):
    """Raised when dataset integrity cannot be restored or foreign data injection persists."""
    pass


@dataclass
class DatasetManifest:
    raw_file_sha256: str
    canonical_train_sha256: str
    content_row_hash: str
    total_rows: int
    allowed_indices: List[int]
    allowed_columns: List[str]


class DataProvenanceGuard:
    """Validates dataset provenance and enforces strict isolation against unauthorized data."""

    def __init__(self, manifest_path: Path, canonical_data_path: Path):
        self.manifest_path = Path(manifest_path)
        self.canonical_data_path = Path(canonical_data_path)
        self.manifest: Optional[DatasetManifest] = None
        if self.manifest_path.exists():
            self.load_manifest()

    def load_manifest(self) -> None:
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.manifest = DatasetManifest(**data)

    @staticmethod
    def compute_file_sha256(file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def compute_dataframe_content_hash(df: pd.DataFrame) -> str:
        row_hashes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha256(row_hashes).hexdigest()

    def verify_integrity(self, df: pd.DataFrame) -> bool:
        """Verifies row count, set-theoretic index containment, and content hash."""
        if not self.manifest:
            logger.warning("No manifest found to verify against.")
            return True

        # Check row count
        if len(df) != self.manifest.total_rows:
            logger.error("Row count mismatch: expected %d, got %d", self.manifest.total_rows, len(df))
            return False

        # Check index containment
        df_indices = set(df.index)
        allowed_set = set(self.manifest.allowed_indices)
        if not df_indices.issubset(allowed_set):
            logger.error("Foreign row indices detected in dataset!")
            return False

        # Check content hash
        content_hash = self.compute_dataframe_content_hash(df)
        if content_hash != self.manifest.content_row_hash:
            logger.error("Content row hash mismatch!")
            return False

        return True

    def force_reload_canonical(self) -> pd.DataFrame:
        """Forcibly reloads the verified canonical dataset partition from disk."""
        if not self.canonical_data_path.exists():
            raise UnauthorizedDatasetException(
                f"Canonical dataset file not found at {self.canonical_data_path}!"
            )
        current_hash = self.compute_file_sha256(self.canonical_data_path)
        if self.manifest and current_hash != self.manifest.canonical_train_sha256:
            raise UnauthorizedDatasetException(
                "Canonical data file on disk has been tampered with! SHA-256 mismatch."
            )
        logger.info("Forcibly reloaded canonical dataset from %s", self.canonical_data_path)
        return pd.read_parquet(self.canonical_data_path)

    def assert_strictly_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Audits df; if invalid, logs an alert and returns the restored canonical dataset."""
        if not self.verify_integrity(df):
            logger.critical("SECURITY ALERT: Dataset tampering / unauthorized data source detected!")
            return self.force_reload_canonical()
        return df
