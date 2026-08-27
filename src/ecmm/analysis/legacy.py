from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from ..config import ProjectConfig


def export_legacy_outputs(
    run_dir: str | Path,
    config: ProjectConfig,
    artifact_dir: str | Path,
) -> Path:
    run_dir = Path(run_dir)
    output = run_dir / "legacy"
    output.mkdir(parents=True, exist_ok=True)
    patterns = np.load(Path(artifact_dir) / "patterns.npz", allow_pickle=False)
    name = config.io.run_name
    with h5py.File(run_dir / "run.h5", "r") as store:
        spike_export = _write_spikes(
            store, output / f"spikes3-{name}.dat", patterns, config
        )
        _write_rates(store, output / f"rate3-{name}.dat")
        _save_table(store, "statistics/window", output / f"temp3-{name}.dat")
        _save_table(store, "statistics/cumulative", output / f"medie3-{name}.dat")
        _write_overlap(store, output / f"q3-{name}.dat", config)
    if config.execution.playback_ms > 0:
        _write_pattern_playback(output, patterns, config)
    (output / "export_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "spikes": spike_export,
                "max_spikes_semantics": (
                    "io.max_spikes limits only the legacy spikes3 text export; "
                    "run.h5 retains every recorded spike"
                ),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return output


def _write_spikes(store, path: Path, patterns, config: ProjectConfig) -> dict[str, int | bool]:
    observed = min(config.monitors.patterns_observed, config.network.patterns)
    posix = patterns["posix"]
    limit = min(len(store["spikes/time_ms"]), config.io.max_spikes)
    with path.open("w", encoding="utf-8") as stream:
        for start in range(0, limit, 65536):
            stop = min(limit, start + 65536)
            times = store["spikes/time_ms"][start:stop]
            neurons = store["spikes/neuron"][start:stop]
            cues = store["spikes/cue"][start:stop]
            for time_ms, neuron, cue in zip(times, neurons, cues):
                positions = "".join(f" {int(posix[p, neuron]):6d}" for p in range(observed))
                stream.write(f"{float(time_ms):20.15g} {int(cue):4d} {int(neuron):6d}{positions}\n")
    return {
        "recorded_hdf5": int(len(store["spikes/time_ms"])),
        "exported_legacy": int(limit),
        "legacy_limit": int(config.io.max_spikes),
        "truncated": bool(limit < len(store["spikes/time_ms"])),
    }


def _write_rates(store, path: Path) -> None:
    rates = np.asarray(store["rates/module_hz"])
    edges = np.asarray(store["rates/edges_ms"])
    table = np.column_stack((edges[1:1 + len(rates)], rates))
    np.savetxt(path, table, fmt="%12g")


def _save_table(store, source: str, path: Path) -> None:
    values = np.asarray(store[source])
    if values.size == 0:
        path.write_text("", encoding="utf-8")
    else:
        np.savetxt(path, values, fmt="%12g")


def _write_overlap(store, path: Path, config: ProjectConfig) -> None:
    time = np.asarray(store["overlap/time_ms"])
    if len(time) == 0:
        path.write_text("", encoding="utf-8")
        return
    starts = np.asarray(store["overlap/window_start_ms"])
    ends = np.asarray(store["overlap/window_end_ms"])
    spikes = np.asarray(store["overlap/spikes"])
    best = np.asarray(store["overlap/best_pattern"])
    maximum = np.asarray(store["overlap/max_overlap"])
    variance = np.asarray(store["overlap/overlap_variance"])
    periods = np.asarray(store["overlap/best_period_ms"])
    overlaps = np.asarray(store["overlap/values"])
    observed = min(config.monitors.patterns_observed, config.network.patterns)
    rows = []
    for index in range(len(time)):
        row = [
            _sigma_at(config, time[index]), config.runtime.delta, config.runtime.alpha,
            starts[index], ends[index], spikes[index], best[index], maximum[index], variance[index],
        ]
        for pattern in range(observed):
            row.extend((periods[index, pattern], overlaps[index, pattern]))
        rows.append(row)
    np.savetxt(path, np.asarray(rows), fmt="%12g")


def _sigma_at(config: ProjectConfig, time_ms: float) -> float:
    runtime = config.runtime
    if runtime.sigma_min is None or runtime.sigma_max is None:
        return runtime.sigma
    half = runtime.duration_ms / 2.0
    reflected = time_ms if time_ms < half else runtime.duration_ms - time_ms
    return runtime.sigma_min + 2 * (runtime.sigma_max - runtime.sigma_min) * reflected / runtime.duration_ms


def _write_pattern_playback(output: Path, patterns, config: ProjectConfig) -> None:
    period_ms = 1000.0 / config.network.frequency_hz
    where = patterns["where"]
    module_sizes = patterns["Z"]
    pattern_count = min(config.monitors.playback_patterns, config.network.patterns)
    for pattern in range(pattern_count):
        hp = int(patterns["H"][pattern])
        events: list[tuple[float, int]] = []
        index = 0
        while True:
            position = index % hp
            cycle = index // hp
            time_ms = period_ms * (
                float(patterns["phi"][pattern, position]) / (2.0 * np.pi) + cycle
            )
            if time_ms > config.execution.playback_ms:
                break
            events.append((time_ms, int(patterns["who"][pattern, position])))
            index += 1
        spike_path = output / f"spikes0-{pattern}-{config.io.run_name}.dat"
        with spike_path.open("w", encoding="utf-8") as stream:
            for time_ms, neuron in events:
                stream.write(
                    f"{time_ms:12g} {1:4d} {neuron:6d} "
                    f"{int(patterns['posix'][pattern, neuron]):6d}\n"
                )
        bins = int(np.ceil(config.execution.playback_ms / config.runtime.output_bin_ms))
        counts = np.zeros((bins, config.network.modules), dtype=np.int64)
        for time_ms, neuron in events:
            bin_index = min(bins - 1, int(max(0.0, time_ms - 1e-7) /
                                          config.runtime.output_bin_ms))
            counts[bin_index, where[neuron]] += 1
        rates = counts * (1000.0 / config.runtime.output_bin_ms) / module_sizes[None, :]
        np.savetxt(output / f"rate0-{pattern}-{config.io.run_name}.dat", rates, fmt="%12g")
