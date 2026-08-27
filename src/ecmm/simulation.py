"""Compatibility facade for the production runtime package."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import ProjectConfig


def run_simulation(
    config: ProjectConfig,
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    duration_ms: float | None = None,
) -> Path:
    from .runtime.runner import SimulationRunner

    if duration_ms is not None:
        config = replace(config, runtime=replace(config.runtime, duration_ms=duration_ms))
    return SimulationRunner(config, artifact_dir, output_dir).run().output_dir
