"""Online monitor contracts for rates, overlaps and progress."""

from dataclasses import dataclass
from typing import Protocol

from .suite import MonitorSuite, PatternOverlapMonitor, RateMonitor, activity_statistics


@dataclass(frozen=True)
class ProgressSnapshot:
    simulated_ms: float
    duration_ms: float
    wall_seconds: float
    spikes: int
    gpu_memory_used_mib: int | None = None


class Monitor(Protocol):
    def update(self, time_ms, spikes) -> None: ...
    def finalize(self) -> dict: ...


__all__ = [
    "Monitor", "MonitorSuite", "PatternOverlapMonitor", "ProgressSnapshot", "RateMonitor",
    "activity_statistics",
]
