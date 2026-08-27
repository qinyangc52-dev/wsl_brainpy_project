from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Checkpoint:
    step: int
    recorded_spikes: int
    total_spikes: int
    config_hash: str
    artifact_identity: dict[str, str]
    network_state: dict[str, np.ndarray]
    monitor_state: dict[str, np.ndarray]
    complete: bool = False


def save_checkpoint(path: str | Path, checkpoint: Checkpoint) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "step": checkpoint.step,
        "recorded_spikes": checkpoint.recorded_spikes,
        "total_spikes": checkpoint.total_spikes,
        "config_hash": checkpoint.config_hash,
        "artifact_identity": checkpoint.artifact_identity,
        "complete": checkpoint.complete,
        "network_keys": sorted(checkpoint.network_state),
        "monitor_keys": sorted(checkpoint.monitor_state),
    }
    arrays = {f"network__{key}": value for key, value in checkpoint.network_state.items()}
    arrays.update({f"monitor__{key}": value for key, value in checkpoint.monitor_state.items()})
    handle, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata)), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_checkpoint(path: str | Path) -> Checkpoint:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
        network = {key: payload[f"network__{key}"].copy() for key in metadata["network_keys"]}
        monitor = {key: payload[f"monitor__{key}"].copy() for key in metadata["monitor_keys"]}
    return Checkpoint(
        step=int(metadata["step"]),
        recorded_spikes=int(metadata["recorded_spikes"]),
        total_spikes=int(metadata["total_spikes"]),
        config_hash=metadata["config_hash"],
        artifact_identity=dict(metadata["artifact_identity"]),
        network_state=network,
        monitor_state=monitor,
        complete=bool(metadata["complete"]),
    )
