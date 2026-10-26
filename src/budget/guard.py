"""BudgetGuard module: Enforces the strict 15 model-training experiment limit invariant."""

import json
from pathlib import Path
from typing import Optional


class BudgetExhaustedError(RuntimeError):
    """Raised when an experiment execution is attempted beyond the 15-run limit."""
    pass


class BudgetGuard:
    """Hard-stop enforcer and persistent tracker for the 15-fit experiment budget."""

    def __init__(self, ledger_path: Path, max_experiments: int = 15):
        self.ledger_path = Path(ledger_path)
        self.max_experiments = max_experiments

    def current_count(self) -> int:
        """Returns the number of completed experiments recorded in the ledger."""
        if not self.ledger_path.exists():
            return 0
        count = 0
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def is_exhausted(self) -> bool:
        """Checks if the budget has reached or exceeded max_experiments."""
        return self.current_count() >= self.max_experiments

    def register_fit_start(self, experiment_id: int) -> None:
        """Registers the start of a fit-and-validation experiment.
        
        Raises:
            BudgetExhaustedError: If current_count >= max_experiments.
        """
        current = self.current_count()
        if current >= self.max_experiments:
            raise BudgetExhaustedError(
                f"STRICT BUDGET EXCEEDED: Attempted to run experiment {experiment_id}, "
                f"but {current}/{self.max_experiments} runs have already been executed. "
                "Disqualification prevented."
            )
