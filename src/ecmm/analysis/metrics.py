from __future__ import annotations

import numpy as np


def binned_rates(
    spike_times_ms: np.ndarray,
    spike_neurons: np.ndarray,
    where: np.ndarray,
    module_sizes: np.ndarray,
    duration_ms: float,
    bin_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(0.0, duration_ms + bin_ms, bin_ms, dtype=np.float64)
    if edges[-1] < duration_ms:
        edges = np.append(edges, duration_ms)
    counts = np.zeros((len(edges) - 1, len(module_sizes)), dtype=np.int32)
    if spike_times_ms.size:
        bins = np.searchsorted(edges, spike_times_ms, side="right") - 1
        valid = (bins >= 0) & (bins < len(counts))
        modules = where[spike_neurons[valid]]
        np.add.at(counts, (bins[valid], modules), 1)
    widths_s = np.diff(edges)[:, None] / 1000.0
    rates = counts / (module_sizes[None, :] * widths_s)
    return edges, rates.astype(np.float32)


def phase_overlap(
    spike_times_ms: np.ndarray,
    spike_neurons: np.ndarray,
    order: np.ndarray,
    phi: np.ndarray,
    H: np.ndarray,
    *,
    window_ms: float = 200.0,
    period_min_ms: float = 10.0,
    period_max_ms: float = 200.0,
    period_step_ms: float = 5.0,
) -> dict[str, list[float] | float | int]:
    if spike_times_ms.size == 0:
        return {
            "window_start_ms": 0.0, "window_end_ms": 0.0, "spikes": 0,
            "best_period_ms": [0.0] * len(H), "overlap": [0.0] * len(H),
        }
    end = float(spike_times_ms[-1])
    start = max(0.0, end - window_ms)
    mask = spike_times_ms >= start
    times = spike_times_ms[mask]
    neurons = spike_neurons[mask]
    denominator = max(1, len(times))
    periods = np.arange(period_min_ms, period_max_ms + 0.5 * period_step_ms, period_step_ms)
    best_periods: list[float] = []
    overlaps: list[float] = []
    for p, _hp in enumerate(H):
        positions = order[p, neurons]
        active = positions >= 0
        if not np.any(active):
            best_periods.append(0.0)
            overlaps.append(0.0)
            continue
        phases = phi[p, positions[active]]
        active_times = times[active]
        values = []
        for period in periods:
            angles = 2.0 * np.pi * active_times / period - phases
            values.append(float(np.abs(np.exp(1j * angles).sum()) / denominator))
        best = int(np.argmax(values))
        best_periods.append(float(periods[best]))
        overlaps.append(values[best])
    return {
        "window_start_ms": start, "window_end_ms": end, "spikes": int(len(times)),
        "best_period_ms": best_periods, "overlap": overlaps,
    }
