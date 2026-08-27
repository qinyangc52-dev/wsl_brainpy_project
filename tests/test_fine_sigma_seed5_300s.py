from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "run_fine_sigma_seed5_300s.py"
SPEC = importlib.util.spec_from_file_location("run_fine_sigma_seed5_300s", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_scan_has_five_selected_sigma_points_and_five_seeds_each():
    tasks = MODULE.build_tasks()
    assert len(tasks) == 25
    assert sorted({task.sigma for task in tasks}) == list(MODULE.SIGMA_VALUES)
    assert MODULE.SIGMA_VALUES == tuple(
        MODULE.Decimal(value) for value in ("6.85", "6.86", "6.87", "6.90", "6.95")
    )
    for sigma in MODULE.SIGMA_VALUES:
        assert [task.seed for task in tasks if task.sigma == sigma] == list(MODULE.SEEDS)


def test_simulation_command_uses_300s_base_and_matching_artifact_seed():
    task = MODULE.build_tasks()[7]
    artifact = PROJECT / "artifacts" / f"full_seed_{task.seed}"
    command = MODULE.simulation_command(
        PROJECT / "configs" / "full_300s.yaml",
        task,
        artifact,
        PROJECT / "runs" / task.run_name,
    )
    joined = " ".join(command)
    assert "configs\\full_300s.yaml" in joined or "configs/full_300s.yaml" in joined
    assert f"runtime.sigma={task.sigma}" in command
    assert f"seeds.network={task.seed}" in command
    assert f"seeds.dynamics={task.seed}" in command
    assert str(artifact) in command
