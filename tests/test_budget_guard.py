"""Unit tests verifying strict enforcement of the 15-experiment limit."""

import pytest
import json
from pathlib import Path
from src.budget.guard import BudgetGuard, BudgetExhaustedError


def test_budget_guard_under_limit(tmp_path: Path):
    ledger_file = tmp_path / "ledger.jsonl"
    guard = BudgetGuard(ledger_path=ledger_file, max_experiments=15)

    assert guard.current_count() == 0
    assert not guard.is_exhausted()

    # Simulate 5 runs
    with open(ledger_file, "w") as f:
        for i in range(5):
            f.write(json.dumps({"experiment_id": i + 1}) + "\n")

    assert guard.current_count() == 5
    assert not guard.is_exhausted()
    # Should not raise
    guard.register_fit_start(6)


def test_budget_guard_blocks_run_16(tmp_path: Path):
    ledger_file = tmp_path / "ledger.jsonl"
    guard = BudgetGuard(ledger_path=ledger_file, max_experiments=15)

    # Populate 15 runs
    with open(ledger_file, "w") as f:
        for i in range(15):
            f.write(json.dumps({"experiment_id": i + 1}) + "\n")

    assert guard.current_count() == 15
    assert guard.is_exhausted()

    # Attempting run 16 MUST raise BudgetExhaustedError
    with pytest.raises(BudgetExhaustedError) as exc_info:
        guard.register_fit_start(16)

    assert "STRICT BUDGET EXCEEDED" in str(exc_info.value)
