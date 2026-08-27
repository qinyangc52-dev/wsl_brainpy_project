from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def next_spike_delays(
    A: np.ndarray,
    B: np.ndarray,
    threshold: float | np.ndarray = 1.0,
    tau_ms: float = 10.0,
) -> np.ndarray:
    """Analytic legacy `prossimaspike()` result for validation."""

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    threshold = np.broadcast_to(np.asarray(threshold, dtype=np.float64), A.shape)
    result = np.full(A.shape, np.inf, dtype=np.float64)
    result[A - B >= threshold] = 0.0
    candidate = (A - B < threshold) & (A > 0) & (B > 0)
    discriminant = A * A - 4.0 * B * threshold
    candidate &= discriminant > 0
    indices = np.flatnonzero(candidate)
    if len(indices):
        x = (A[indices] + np.sqrt(discriminant[indices])) / (2.0 * B[indices])
        valid = (x > 0) & (x < 1)
        result[indices[valid]] = -tau_ms * np.log(x[valid])
    return result


@dataclass
class LegacyEventReference:
    """Small deterministic event backend used only as a correctness oracle."""

    weights: np.ndarray
    sigma: float
    delta: float
    tau_ms: float = 10.0
    threshold: float = 1.0

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.weights.ndim != 2 or self.weights.shape[0] != self.weights.shape[1]:
            raise ValueError("weights must be a square post_by_pre matrix")
        self.A = np.zeros(self.weights.shape[0], dtype=np.float64)
        self.B = np.zeros(self.weights.shape[0], dtype=np.float64)
        self.time_ms = 0.0

    def advance(self, target_ms: float) -> None:
        if target_ms < self.time_ms:
            raise ValueError("event reference cannot move backwards")
        elapsed = target_ms - self.time_ms
        decay_a = np.exp(-elapsed / self.tau_ms)
        self.A *= decay_a
        self.B *= decay_a * decay_a
        self.time_ms = target_ms

    def inject(self, neuron: int, amplitude: float) -> None:
        self.A[neuron] += amplitude
        self.B[neuron] += amplitude

    def next_internal_spike(self) -> tuple[float, int] | None:
        delays = next_spike_delays(self.A, self.B, self.threshold, self.tau_ms)
        neuron = int(np.argmin(delays))
        if not np.isfinite(delays[neuron]):
            return None
        return self.time_ms + float(delays[neuron]), neuron

    def emit(self, neuron: int) -> None:
        self.A[neuron] = 0.0
        self.B[neuron] = 0.0
        drive = self.sigma * self.weights[:, neuron] - self.delta
        drive[neuron] = 0.0
        self.A += drive
        self.B += drive
