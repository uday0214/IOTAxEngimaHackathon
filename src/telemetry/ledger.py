"""Append-only experiment ledger for auditability and Slide Deck reporting."""

import json
from pathlib import Path
from typing import Dict, List, Optional


class ExperimentLedger:
    """Manages appending and reading runs in artifacts/experiment_ledger.jsonl."""

    def __init__(self, ledger_path: Path = Path("artifacts/experiment_ledger.jsonl")):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def append_run(self, entry: Dict) -> None:
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def load_all_runs(self) -> List[Dict]:
        if not self.ledger_path.exists():
            return []
        runs = []
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    runs.append(json.loads(line))
        return runs

    def get_best_run(self) -> Optional[Dict]:
        runs = self.load_all_runs()
        if not runs:
            return None
        return max(runs, key=lambda r: r.get("metrics", {}).get("oof_roc_auc", 0.0))

    # Backward compatibility aliases
    append_record = append_run
    load_history = load_all_runs

