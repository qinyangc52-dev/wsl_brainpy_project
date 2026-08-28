from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np

from ..config import CueConfig, RuntimeConfig


@dataclass(frozen=True)
class NoiseInput:
    """Counter-based Poisson input independent of chunk boundaries."""

    seed: int
    alpha: float
    rho_per_ms: float
    dt_ms: float
    mode: str = "gaussian"

    @classmethod
    def from_config(cls, seed: int, runtime: RuntimeConfig) -> "NoiseInput":
        return cls(seed, runtime.alpha, runtime.rho, runtime.dt_ms, runtime.noise_mode)

    def chunk(self, start_step: int, steps: int, neurons: int):
        if steps <= 0:
            return jnp.empty((0, neurons), dtype=jnp.float32)
        step_ids = jnp.arange(start_step, start_step + steps, dtype=jnp.uint32)
        base_key = jax.random.PRNGKey(self.seed)

        def sample(step_id):
            key = jax.random.fold_in(base_key, step_id)
            count_key, amplitude_key = jax.random.split(key)
            counts = jax.random.poisson(
                count_key, lam=self.rho_per_ms * self.dt_ms, shape=(neurons,)
            ).astype(jnp.float32)
            if self.mode == "constant":
                return self.alpha * counts
            if self.mode != "gaussian":
                raise ValueError(f"Unsupported noise mode: {self.mode}")
            amplitudes = jax.random.normal(amplitude_key, (neurons,), dtype=jnp.float32)
            return self.alpha * jnp.sqrt(counts) * amplitudes

        return jax.vmap(sample)(step_ids)


@dataclass(frozen=True)
class CueInput:
    """Discrete forced-spike schedule generated from legacy cue descriptions."""

    steps: np.ndarray
    neurons: np.ndarray
    total_neurons: int

    @classmethod
    def from_patterns(
        cls,
        cues: tuple[CueConfig, ...],
        who: np.ndarray,
        H: np.ndarray,
        dt_ms: float,
        total_neurons: int,
    ) -> "CueInput":
        event_steps: list[int] = []
        event_neurons: list[int] = []
        for cue in cues:
            hp = int(H[cue.pattern])
            if hp <= 0:
                continue
            for index in range(cue.spike_count):
                time_ms = cue.start_ms + (1000.0 / cue.frequency_hz) * index / hp
                event_steps.append(max(0, int(round(time_ms / dt_ms)) - 1))
                event_neurons.append(int(who[cue.pattern, index % hp]))
        if not event_steps:
            return cls(np.empty(0, np.int64), np.empty(0, np.int32), total_neurons)
        order = np.argsort(event_steps, kind="stable")
        return cls(
            np.asarray(event_steps, dtype=np.int64)[order],
            np.asarray(event_neurons, dtype=np.int32)[order],
            total_neurons,
        )

    def chunk(self, start_step: int, steps: int) -> np.ndarray:
        result = np.zeros((steps, self.total_neurons), dtype=bool)
        left = np.searchsorted(self.steps, start_step, side="left")
        right = np.searchsorted(self.steps, start_step + steps, side="left")
        local_steps = self.steps[left:right] - start_step
        result[local_steps, self.neurons[left:right]] = True
        return result


@dataclass(frozen=True)
class SigmaScheduler:
    sigma: float
    duration_ms: float
    dt_ms: float
    sigma_min: float | None = None
    sigma_max: float | None = None

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.sigma, self.duration_ms, self.dt_ms)
        ):
            raise ValueError("sigma, duration_ms and dt_ms must be finite")
        if self.duration_ms <= 0 or self.dt_ms <= 0:
            raise ValueError("duration_ms and dt_ms must be positive")
        if (self.sigma_min is None) != (self.sigma_max is None):
            raise ValueError("sigma_min and sigma_max must be set together")
        if self.sigma_min is not None and self.sigma_max is not None:
            if not math.isfinite(self.sigma_min) or not math.isfinite(self.sigma_max):
                raise ValueError("sigma bounds must be finite")
            if self.sigma_min > self.sigma_max:
                raise ValueError("sigma_min cannot exceed sigma_max")

    @classmethod
    def from_config(cls, runtime: RuntimeConfig) -> "SigmaScheduler":
        return cls(
            runtime.sigma,
            runtime.duration_ms,
            runtime.dt_ms,
            runtime.sigma_min,
            runtime.sigma_max,
        )

    @property
    def varying(self) -> bool:
        return self.sigma_min is not None and self.sigma_max is not None

    def values(self, start_step: int, steps: int):
        if not self.varying:
            return jnp.full((steps,), self.sigma, dtype=jnp.float32)
        times = (jnp.arange(start_step, start_step + steps, dtype=jnp.float32) + 1) * self.dt_ms
        half = self.duration_ms / 2.0
        reflected = jnp.where(times < half, times, self.duration_ms - times)
        slope = 2.0 * (self.sigma_max - self.sigma_min) / self.duration_ms
        return (self.sigma_min + slope * reflected).astype(jnp.float32)
