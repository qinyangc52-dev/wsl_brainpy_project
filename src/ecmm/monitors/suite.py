from __future__ import annotations

import json
import math

import numpy as np

from ..analysis import phase_overlap
from ..config import ProjectConfig
from ..models import SigmaScheduler


class RateMonitor:
    def __init__(self, config: ProjectConfig, where: np.ndarray, module_sizes: np.ndarray):
        self.bin_ms = config.runtime.output_bin_ms
        self.duration_ms = config.runtime.duration_ms
        self.where = np.asarray(where, dtype=np.int32)
        self.module_sizes = np.asarray(module_sizes, dtype=np.int32)
        bins = int(math.ceil(self.duration_ms / self.bin_ms))
        self.counts = np.zeros((bins, len(module_sizes)), dtype=np.int64)

    def update(self, times_ms: np.ndarray, neurons: np.ndarray) -> None:
        if len(times_ms) == 0:
            return
        bins = np.floor((times_ms - 1e-7) / self.bin_ms).astype(np.int64)
        modules = self.where[neurons]
        valid = (bins >= 0) & (bins < len(self.counts))
        np.add.at(self.counts, (bins[valid], modules[valid]), 1)

    def module_rates(self, counts: np.ndarray | None = None) -> np.ndarray:
        counts = self.counts if counts is None else counts
        return (counts * (1000.0 / self.bin_ms) / self.module_sizes[None, :]).astype(np.float32)


class PatternOverlapMonitor:
    def __init__(self, config: ProjectConfig, patterns):
        self.config = config
        self.order = np.asarray(patterns["order"])
        self.phi = np.asarray(patterns["phi"])
        self.H = np.asarray(patterns["H"])
        self.flush_ms = config.monitors.flush_bins * config.runtime.output_bin_ms
        self.next_flush_ms = self.flush_ms
        self.recent_times = np.empty(0, dtype=np.float32)
        self.recent_neurons = np.empty(0, dtype=np.int32)
        self.rows: list[dict] = []

    def update(self, times_ms: np.ndarray, neurons: np.ndarray, chunk_end_ms: float) -> None:
        if len(times_ms):
            self.recent_times = np.concatenate((self.recent_times, times_ms.astype(np.float32)))
            self.recent_neurons = np.concatenate((self.recent_neurons, neurons.astype(np.int32)))
        while self.next_flush_ms <= chunk_end_ms + 1e-7:
            eligible = ((self.recent_times <= self.next_flush_ms + 1e-7) &
                        (self.recent_times >= self.config.monitors.overlap_start_ms))
            selected_times = self.recent_times[eligible]
            selected_neurons = self.recent_neurons[eligible]
            available_window_ms = min(
                self.config.monitors.overlap_window_ms,
                max(0.0, self.next_flush_ms - self.config.monitors.overlap_start_ms),
            )
            result = phase_overlap(
                selected_times, selected_neurons, self.order, self.phi, self.H,
                window_ms=available_window_ms,
                window_end_ms=self.next_flush_ms,
            )
            covered = result["window_end_ms"] - result["window_start_ms"]
            if (result["spikes"] <= 2 or
                    covered + 1e-8 < self.config.monitors.overlap_min_fraction *
                    self.config.monitors.overlap_window_ms):
                result["best_period_ms"] = [0.0] * len(self.H)
                result["overlap"] = [0.0] * len(self.H)
            overlaps = result["overlap"]
            best_pattern = int(np.argmax(overlaps)) if overlaps else 0
            self.rows.append(
                {
                    "time_ms": self.next_flush_ms,
                    "window_start_ms": result["window_start_ms"],
                    "window_end_ms": result["window_end_ms"],
                    "spikes": result["spikes"],
                    "best_pattern": best_pattern,
                    "max_overlap": overlaps[best_pattern] if overlaps else 0.0,
                    "best_period_ms": result["best_period_ms"],
                    "overlap": overlaps,
                }
            )
            self.next_flush_ms += self.flush_ms
        keep_after = max(0.0, self.next_flush_ms - self.flush_ms - self.config.monitors.overlap_window_ms)
        keep = self.recent_times >= keep_after
        self.recent_times = self.recent_times[keep]
        self.recent_neurons = self.recent_neurons[keep]

    def snapshot(self) -> dict[str, np.ndarray]:
        return {
            "overlap_recent_times": self.recent_times,
            "overlap_recent_neurons": self.recent_neurons,
            "overlap_next_flush": np.asarray(self.next_flush_ms),
            "overlap_rows_json": np.asarray(json.dumps(self.rows)),
        }

    def restore(self, state: dict[str, np.ndarray]) -> None:
        self.recent_times = state["overlap_recent_times"].copy()
        self.recent_neurons = state["overlap_recent_neurons"].copy()
        self.next_flush_ms = float(state["overlap_next_flush"])
        self.rows = json.loads(str(state["overlap_rows_json"]))


class MonitorSuite:
    def __init__(self, config: ProjectConfig, patterns):
        self.config = config
        self.rate = RateMonitor(config, patterns["where"], patterns["Z"])
        self.overlap = PatternOverlapMonitor(config, patterns)

    def update_chunk(
        self,
        times_ms: np.ndarray,
        neurons: np.ndarray,
        chunk_end_ms: float,
    ) -> None:
        self.rate.update(times_ms, neurons)
        self.overlap.update(times_ms, neurons, chunk_end_ms)

    def snapshot(self) -> dict[str, np.ndarray]:
        return {"rate_counts": self.rate.counts, **self.overlap.snapshot()}

    def restore(self, state: dict[str, np.ndarray]) -> None:
        if state["rate_counts"].shape != self.rate.counts.shape:
            raise ValueError("Checkpoint rate monitor shape does not match configuration")
        self.rate.counts[...] = state["rate_counts"]
        self.overlap.restore(state)

    def latest_rate_hz(self, completed_step: int) -> float:
        completed_ms = completed_step * self.config.runtime.dt_ms
        completed_bins = min(len(self.rate.counts), int(completed_ms / self.rate.bin_ms))
        window = self.config.monitors.flush_bins
        if completed_bins == 0:
            return 0.0
        counts = self.rate.counts[max(0, completed_bins - window):completed_bins].sum()
        elapsed_ms = min(window, completed_bins) * self.rate.bin_ms
        return 1000.0 * counts / (self.config.network.total_neurons * elapsed_ms)

    def flush_window_rate_hz(self, window_index: int) -> float:
        flush = self.config.monitors.flush_bins
        stop = window_index * flush
        start = stop - flush
        if start < 0 or stop > len(self.rate.counts):
            raise IndexError("flush window is outside the rate monitor")
        counts = self.rate.counts[start:stop].sum()
        elapsed_ms = flush * self.rate.bin_ms
        return 1000.0 * counts / (self.config.network.total_neurons * elapsed_ms)

    def finalize(self, store, completed_ms: float | None = None) -> dict[str, np.ndarray]:
        bins = len(self.rate.counts)
        if completed_ms is not None:
            bins = min(bins, int(math.ceil(completed_ms / self.rate.bin_ms)))
        counts = self.rate.counts[:bins]
        edges = np.arange(len(counts) + 1, dtype=np.float64) * self.rate.bin_ms
        rates = self.rate.module_rates(counts)
        temp, cumulative = activity_statistics(self.config, counts)
        store.write_dataset("rates/edges_ms", edges)
        store.write_dataset("rates/counts", counts)
        store.write_dataset("rates/module_hz", rates)
        store.write_dataset("statistics/window", temp)
        store.write_dataset("statistics/cumulative", cumulative)
        overlap = overlap_arrays(
            self.overlap.rows,
            self.config.network.patterns,
            self.config.monitors.overlap_history,
        )
        for name, values in overlap.items():
            store.write_dataset(f"overlap/{name}", values)
        store.flush()
        return {"edges_ms": edges, "module_hz": rates, "window": temp,
                "cumulative": cumulative, **{f"overlap_{k}": v for k, v in overlap.items()}}


def _sigma_at(config: ProjectConfig, time_ms: float) -> float:
    runtime = config.runtime
    if runtime.sigma_min is None or runtime.sigma_max is None:
        return runtime.sigma
    half = runtime.duration_ms / 2.0
    reflected = time_ms if time_ms < half else runtime.duration_ms - time_ms
    return runtime.sigma_min + 2.0 * (runtime.sigma_max - runtime.sigma_min) * reflected / runtime.duration_ms


def activity_statistics(config: ProjectConfig, module_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    global_counts = module_counts.sum(axis=1).astype(np.float64)
    flush = config.monitors.flush_bins
    temp_rows = []
    cumulative_rows = []
    for stop in range(flush, len(global_counts) + 1, flush):
        time_ms = stop * config.runtime.output_bin_ms
        for source, target in ((global_counts[stop - flush:stop], temp_rows),
                               (global_counts[:stop], cumulative_rows)):
            mean = float(source.mean()) if len(source) else 0.0
            var = float(source.var()) if len(source) else 0.0
            rate = 1000.0 * mean / (config.network.total_neurons * config.runtime.output_bin_ms)
            cv = math.sqrt(max(0.0, var)) / mean if mean > 0 else 0.0
            fano = var / mean if mean > 0 else 0.0
            target.append([
                _sigma_at(config, time_ms), config.runtime.delta, config.runtime.alpha,
                time_ms, rate, var, cv, fano,
            ])
    return np.asarray(temp_rows, np.float64), np.asarray(cumulative_rows, np.float64)


def overlap_arrays(rows: list[dict], patterns: int, history_length: int = 20) -> dict[str, np.ndarray]:
    if not rows:
        return {
            "time_ms": np.empty(0), "window_start_ms": np.empty(0),
            "window_end_ms": np.empty(0), "spikes": np.empty(0, np.int64),
            "best_pattern": np.empty(0, np.int32), "max_overlap": np.empty(0),
            "overlap_variance": np.empty(0),
            "best_period_ms": np.empty((0, patterns)), "values": np.empty((0, patterns)),
        }
    max_values = np.asarray([row["max_overlap"] for row in rows])
    history = []
    # Match qstack semantics: variance of recent maximum overlaps.
    for index in range(len(max_values)):
        start = max(0, index + 1 - history_length)
        history.append(float(np.var(max_values[start:index + 1])))
    return {
        "time_ms": np.asarray([row["time_ms"] for row in rows]),
        "window_start_ms": np.asarray([row["window_start_ms"] for row in rows]),
        "window_end_ms": np.asarray([row["window_end_ms"] for row in rows]),
        "spikes": np.asarray([row["spikes"] for row in rows], np.int64),
        "best_pattern": np.asarray([row["best_pattern"] for row in rows], np.int32),
        "max_overlap": max_values,
        "overlap_variance": np.asarray(history),
        "best_period_ms": np.asarray([row["best_period_ms"] for row in rows]),
        "values": np.asarray([row["overlap"] for row in rows]),
    }
