from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "max"

    def __post_init__(self) -> None:
        if self.mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.best = None
        self.num_bad = 0

    def step(self, value: float) -> bool:
        """
        Returns True if should stop.
        """
        if self.best is None:
            self.best = value
            self.num_bad = 0
            return False

        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.num_bad = 0
            return False

        self.num_bad += 1
        return self.num_bad >= self.patience

