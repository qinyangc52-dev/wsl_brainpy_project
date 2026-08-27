from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


class RunStore:
    """Append-only HDF5 store with checkpoint-aligned truncation."""

    def __init__(self, path: str | Path, *, create: bool, metadata: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if create else "r+"
        self.file = h5py.File(self.path, mode)
        if create:
            spikes = self.file.create_group("spikes")
            spikes.create_dataset("time_ms", shape=(0,), maxshape=(None,), dtype="f4", chunks=True)
            spikes.create_dataset("neuron", shape=(0,), maxshape=(None,), dtype="i4", chunks=True)
            spikes.create_dataset("cue", shape=(0,), maxshape=(None,), dtype="?", chunks=True)
            self.file.attrs["metadata"] = json.dumps(metadata or {}, sort_keys=True)
            self.file.flush()

    @property
    def spike_count(self) -> int:
        return int(self.file["spikes/time_ms"].shape[0])

    @property
    def metadata(self) -> dict:
        return json.loads(self.file.attrs.get("metadata", "{}"))

    def append_spikes(self, times: np.ndarray, neurons: np.ndarray, cues: np.ndarray) -> None:
        count = len(times)
        if count == 0:
            return
        start = self.spike_count
        stop = start + count
        for name, values in (("time_ms", times), ("neuron", neurons), ("cue", cues)):
            dataset = self.file[f"spikes/{name}"]
            dataset.resize((stop,))
            dataset[start:stop] = values
        self.file.flush()

    def truncate_spikes(self, count: int) -> None:
        if count < 0 or count > self.spike_count:
            raise ValueError("Invalid spike truncation point")
        for name in ("time_ms", "neuron", "cue"):
            self.file[f"spikes/{name}"].resize((count,))
        self.file.flush()

    def write_dataset(self, path: str, values: np.ndarray) -> None:
        if path in self.file:
            del self.file[path]
        parent, _, name = path.rpartition("/")
        group = self.file.require_group(parent) if parent else self.file
        group.create_dataset(name, data=values, compression="gzip", compression_opts=4)

    def read_spikes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(np.asarray(self.file[f"spikes/{name}"]) for name in ("time_ms", "neuron", "cue"))

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        if self.file:
            self.file.close()

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
