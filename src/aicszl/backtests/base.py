from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .dataset import BacktestDatasetArtifact


@dataclass(frozen=True)
class BacktestRunSettings:
    topk: int
    n_drop: int
    initial_cash: float
    benchmark: str | None = None

    def __post_init__(self) -> None:
        if self.topk < 1:
            raise ValueError("topk must be at least 1")
        if self.n_drop < 1 or self.n_drop > self.topk:
            raise ValueError("n_drop must be between 1 and topk")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.benchmark is not None:
            raise ValueError("Task 7 Qlib POC supports benchmark=None only")


@dataclass(frozen=True)
class BacktestRunArtifact:
    engine: str
    report_path: Path
    positions_path: Path


@runtime_checkable
class BacktestAdapter(Protocol):
    def run(
        self,
        dataset: BacktestDatasetArtifact,
        settings: BacktestRunSettings,
    ) -> BacktestRunArtifact: ...
