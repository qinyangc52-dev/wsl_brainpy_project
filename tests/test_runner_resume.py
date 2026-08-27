from dataclasses import replace
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from ecmm.artifacts import resolve_run_artifact
from ecmm.config import load_config
from ecmm.runtime import SimulationRunner, load_checkpoint
import ecmm.runtime.runner as runner_module


PROJECT = Path(__file__).resolve().parents[1]
ARTIFACT = PROJECT / "artifacts" / "prototype_seed_1256878"


def short_config():
    base = load_config(PROJECT / "configs" / "prototype.yaml")
    return replace(
        base,
        runtime=replace(base.runtime, duration_ms=4.0, chunk_ms=2.0),
        execution=replace(base.execution, progress_interval_s=0.0),
    )


def read_spikes(path):
    with h5py.File(path / "run.h5", "r") as store:
        return tuple(np.asarray(store[f"spikes/{key}"]) for key in ("time_ms", "neuron", "cue"))


def test_pause_resume_matches_uninterrupted_run(tmp_path):
    config = short_config()
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    assert SimulationRunner(config, ARTIFACT, full).run().status == "complete"
    paused_result = SimulationRunner(config, ARTIFACT, resumed).run(max_chunks=1)
    assert paused_result.status == "paused"
    checkpoint = load_checkpoint(resumed / "checkpoint.npz")
    assert checkpoint.step == 20
    assert not checkpoint.complete
    resumed_result = SimulationRunner(config, ARTIFACT, resumed).run(resume=True)
    assert resumed_result.status == "complete"
    for expected, actual in zip(read_spikes(full), read_spikes(resumed)):
        np.testing.assert_array_equal(actual, expected)
    full_state = load_checkpoint(full / "checkpoint.npz")
    resumed_state = load_checkpoint(resumed / "checkpoint.npz")
    for key in full_state.network_state:
        np.testing.assert_array_equal(resumed_state.network_state[key], full_state.network_state[key])


def test_resume_rejects_changed_configuration(tmp_path):
    config = short_config()
    output = tmp_path / "run"
    SimulationRunner(config, ARTIFACT, output).run(max_chunks=1)
    changed = replace(config, runtime=replace(config.runtime, sigma=config.runtime.sigma + 0.1))
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        SimulationRunner(changed, ARTIFACT, output).run(resume=True)


def test_completed_run_contains_streamed_monitor_datasets(tmp_path):
    output = tmp_path / "run"
    SimulationRunner(short_config(), ARTIFACT, output).run()
    with h5py.File(output / "run.h5", "r") as store:
        for path in (
            "spikes/time_ms", "spikes/neuron", "spikes/cue", "rates/module_hz",
            "statistics/window", "statistics/cumulative", "overlap/values",
        ):
            assert path in store
    assert load_checkpoint(output / "checkpoint.npz").complete


def test_fresh_run_refuses_to_overwrite_existing_store(tmp_path):
    output = tmp_path / "run"
    SimulationRunner(short_config(), ARTIFACT, output).run(max_chunks=1)
    with pytest.raises(FileExistsError):
        SimulationRunner(short_config(), ARTIFACT, output).run()


def test_artifact_integrity_is_recomputed_not_only_trusted_from_manifest(tmp_path):
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    with (copied / "connectivity_csr.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="integrity check failed"):
        SimulationRunner(short_config(), copied, tmp_path / "run")


def test_artifact_is_rejected_when_network_structure_does_not_match():
    config = short_config()
    changed = replace(
        config,
        network=replace(config.network, swap=config.network.swap + 0.25),
    )
    with pytest.raises(ValueError, match="structural hash mismatch"):
        SimulationRunner(changed, ARTIFACT, Path("unused"))


def test_relative_artifact_path_survives_project_relocation(tmp_path):
    source = tmp_path / "source"
    artifact = source / "artifacts" / ARTIFACT.name
    run = source / "runs" / "relocated"
    shutil.copytree(ARTIFACT, artifact)
    config = short_config()
    SimulationRunner(config, artifact, run).run(max_chunks=1)

    moved = tmp_path / "moved"
    shutil.move(source, moved)
    moved_run = moved / "runs" / "relocated"
    import json

    manifest = json.loads((moved_run / "run_manifest.json").read_text(encoding="utf-8"))
    resolved = resolve_run_artifact(moved_run, config, manifest)
    assert resolved == (moved / "artifacts" / ARTIFACT.name).resolve()
    assert SimulationRunner(config, resolved, moved_run).run(resume=True).status == "complete"


def test_checkpoint_interval_controls_periodic_writes(tmp_path, monkeypatch):
    calls = []
    real_save = runner_module.save_checkpoint

    def tracking_save(path, checkpoint):
        calls.append(checkpoint.step)
        real_save(path, checkpoint)

    monkeypatch.setattr(runner_module, "save_checkpoint", tracking_save)
    config = short_config()
    SimulationRunner(config, ARTIFACT, tmp_path / "long-interval").run()
    assert calls == [0, 40]


def test_zero_checkpoint_interval_saves_every_chunk(tmp_path, monkeypatch):
    calls = []
    real_save = runner_module.save_checkpoint

    def tracking_save(path, checkpoint):
        calls.append(checkpoint.step)
        real_save(path, checkpoint)

    monkeypatch.setattr(runner_module, "save_checkpoint", tracking_save)
    config = short_config()
    config = replace(
        config,
        execution=replace(config.execution, checkpoint_interval_s=0.0),
    )
    SimulationRunner(config, ARTIFACT, tmp_path / "every-chunk").run()
    assert calls == [0, 20, 40, 40]
