from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Sequence


class ConfigError(ValueError):
    """Raised when an ECMM configuration violates the project contract."""


SizeSpec = int | tuple[int, ...]


@dataclass(frozen=True)
class NetworkConfig:
    topology: str = "random"
    modules: int = 20
    neurons_per_module: SizeSpec = 100
    active_modules_per_pattern: int = 10
    active_neurons_per_module: SizeSpec = 50
    patterns: int = 1
    sort: int = 0
    swap: float = 0.0
    range: float = 1.0
    frequency_hz: float = 8.0
    ddec: float = 3.3
    dmax: float = 30.0
    pmax: int = 3

    def module_sizes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (
            _expand_size(self.neurons_per_module, self.modules, "neurons_per_module"),
            _expand_size(
                self.active_neurons_per_module,
                self.modules,
                "active_neurons_per_module",
            ),
        )

    @property
    def total_neurons(self) -> int:
        return sum(self.module_sizes()[0])


@dataclass(frozen=True)
class RuntimeConfig:
    sigma: float = 1.0
    delta: float = 0.0
    alpha: float = 1.0
    rho: float = 1.0
    noise_mode: str = "gaussian"
    dt_ms: float = 0.1
    output_bin_ms: float = 1.0
    duration_ms: float = 1000.0
    chunk_ms: float = 100.0
    tau_a_ms: float = 10.0
    tau_b_ms: float = 5.0
    threshold: float = 1.0
    sigma_min: float | None = None
    sigma_max: float | None = None


@dataclass(frozen=True)
class MonitorConfig:
    flush_bins: int = 10
    overlap_history: int = 20
    overlap_start_ms: float = 100.0
    overlap_window_ms: float = 200.0
    overlap_min_fraction: float = 0.5
    patterns_observed: int = 3
    playback_patterns: int = 5


@dataclass(frozen=True)
class ExecutionConfig:
    progress_interval_s: float = 30.0
    stop_rate_hz: float = 0.0
    stop_windows: int = 0
    playback_ms: float = 0.0
    cpu_limit_s: float = 0.0
    checkpoint_interval_s: float = 3600.0
    worker_index: int = 0
    debug_level: int = 1


@dataclass(frozen=True)
class SeedConfig:
    network: int = 0
    stream: int = 0
    offline: int = 0
    dynamics: int = 0

    @property
    def effective_offline(self) -> int:
        return self.offline or self.network

    @property
    def effective_dynamics(self) -> int:
        return self.dynamics or self.network


@dataclass(frozen=True)
class ArtifactConfig:
    name: str = "network"
    dtype: str = "float32"
    stdp_block_size: int = 256


@dataclass(frozen=True)
class IOConfig:
    run_name: str = "new"
    temp_dir: str = "output"
    output_dir: str | None = None
    legacy_file_mode: int = 0
    max_spikes: int = 1 << 20
    stdout_path: str | None = None

    @property
    def effective_output_dir(self) -> str:
        return self.output_dir or self.temp_dir


@dataclass(frozen=True)
class CueConfig:
    pattern: int
    start_ms: float
    spike_count: int
    frequency_hz: float


@dataclass(frozen=True)
class ProjectConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    monitors: MonitorConfig = field(default_factory=MonitorConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    io: IOConfig = field(default_factory=IOConfig)
    cues: tuple[CueConfig, ...] = ()
    legacy_unmapped: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> ProjectConfig:
        errors = validate_config(self)
        if errors:
            raise ConfigError("Invalid ECMM configuration:\n- " + "\n- ".join(errors))
        return self


def _expand_size(value: SizeSpec, modules: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * modules
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(int(item) for item in value)
        if len(result) != modules:
            raise ConfigError(f"{name} must contain exactly {modules} values")
        return result
    raise ConfigError(f"{name} must be an integer or a sequence of integers")


def validate_config(config: ProjectConfig) -> list[str]:
    errors: list[str] = []
    n = config.network
    r = config.runtime
    m = config.monitors
    e = config.execution
    io = config.io
    artifact = config.artifact
    if n.modules <= 0:
        errors.append("network.modules must be positive")
    if n.active_modules_per_pattern <= 0 or n.active_modules_per_pattern > n.modules:
        errors.append("network.active_modules_per_pattern must be in [1, modules]")
    if n.patterns <= 0:
        errors.append("network.patterns must be positive")
    if n.sort not in (0, 1, 2):
        errors.append("network.sort must be 0, 1 or 2")
    if n.swap < 0 or n.range < 0:
        errors.append("network.swap and network.range must be non-negative")
    if n.frequency_hz <= 0 or n.ddec <= 0 or n.dmax <= 0:
        errors.append("network frequency_hz, ddec and dmax must be positive")
    if n.topology == "tract1" and n.modules != 66:
        errors.append("network.topology=tract1 requires exactly 66 modules")
    if n.topology != "random" and not n.topology.startswith("tract"):
        errors.append("network.topology must be random or start with tract")
    try:
        z, k = n.module_sizes()
        if any(value <= 0 for value in z):
            errors.append("all module neuron counts must be positive")
        if any(value <= 0 for value in k):
            errors.append("all active neuron counts must be positive")
        if any(active > total for active, total in zip(k, z)):
            errors.append("active neurons per module cannot exceed module size")
    except ConfigError as exc:
        errors.append(str(exc))
    if min(r.dt_ms, r.output_bin_ms, r.duration_ms, r.chunk_ms) <= 0:
        errors.append("runtime time values must be positive")
    if min(r.tau_a_ms, r.tau_b_ms, r.threshold) <= 0:
        errors.append("runtime time constants and threshold must be positive")
    if r.alpha < 0 or r.rho < 0 or r.delta < 0:
        errors.append("runtime alpha, rho and delta must be non-negative")
    runtime_numbers = (
        r.sigma, r.delta, r.alpha, r.rho, r.dt_ms, r.output_bin_ms, r.duration_ms,
        r.chunk_ms, r.tau_a_ms, r.tau_b_ms, r.threshold,
    )
    if not all(math.isfinite(value) for value in runtime_numbers):
        errors.append("runtime numeric values must be finite")
    if r.noise_mode not in ("gaussian", "constant"):
        errors.append("runtime.noise_mode must be gaussian or constant")
    if r.chunk_ms + 1e-12 < r.dt_ms or r.output_bin_ms + 1e-12 < r.dt_ms:
        errors.append("runtime chunk/output bin cannot be smaller than dt")
    if math.isfinite(r.dt_ms) and r.dt_ms > 0:
        for name, value in (
            ("output_bin_ms", r.output_bin_ms),
            ("duration_ms", r.duration_ms),
            ("chunk_ms", r.chunk_ms),
        ):
            if math.isfinite(value) and not _is_step_aligned(value, r.dt_ms):
                errors.append(f"runtime.{name} must be an integer multiple of runtime.dt_ms")
    sigma_bounds = (r.sigma_min, r.sigma_max)
    if (r.sigma_min is None) != (r.sigma_max is None):
        errors.append("runtime.sigma_min and sigma_max must be set together")
    elif all(value is not None for value in sigma_bounds):
        sigma_min, sigma_max = sigma_bounds
        if not math.isfinite(sigma_min) or not math.isfinite(sigma_max):
            errors.append("runtime sigma bounds must be finite")
        elif sigma_min > sigma_max:
            errors.append("runtime.sigma_min cannot exceed sigma_max")
    if min(m.flush_bins, m.overlap_history, m.patterns_observed, m.playback_patterns) <= 0:
        errors.append("monitor integer windows/counts must be positive")
    if m.overlap_start_ms < 0 or m.overlap_window_ms <= 0:
        errors.append("monitor overlap times are invalid")
    if not 0 < m.overlap_min_fraction <= 1:
        errors.append("monitors.overlap_min_fraction must be in (0, 1]")
    if min(e.progress_interval_s, e.stop_rate_hz, e.stop_windows, e.playback_ms,
           e.cpu_limit_s, e.checkpoint_interval_s) < 0:
        errors.append("execution values must be non-negative")
    if e.worker_index < 0 or e.debug_level < 0:
        errors.append("execution worker_index and debug_level must be non-negative")
    if io.legacy_file_mode not in range(8):
        errors.append("io.legacy_file_mode must be a bit mask from 0 to 7")
    if io.max_spikes <= 0:
        errors.append("io.max_spikes must be positive")
    if (not isinstance(artifact.stdp_block_size, int) or
            isinstance(artifact.stdp_block_size, bool) or
            artifact.stdp_block_size <= 0):
        errors.append("artifact.stdp_block_size must be a positive integer")
    for name, value in asdict(config.seeds).items():
        if value < 0:
            errors.append(f"seeds.{name} must be non-negative")
    for index, cue in enumerate(config.cues):
        if cue.pattern < 0 or cue.pattern >= n.patterns:
            errors.append(f"cues[{index}].pattern is outside configured patterns")
        if cue.start_ms < 0 or cue.spike_count <= 0 or cue.frequency_hz <= 0:
            errors.append(f"cues[{index}] has invalid time/count/frequency")
    return errors


def _is_step_aligned(value: float, dt_ms: float) -> bool:
    nearest = round(value / dt_ms)
    tolerance = 1e-9 * max(1.0, abs(value), abs(dt_ms))
    return abs(value - nearest * dt_ms) <= tolerance


def exact_step_count(value_ms: float, dt_ms: float, name: str = "time interval") -> int:
    """Convert an already configured time interval without silent quantization."""
    if (not math.isfinite(value_ms) or not math.isfinite(dt_ms) or
            value_ms <= 0 or dt_ms <= 0):
        raise ConfigError(f"{name} and runtime.dt_ms must be finite and positive")
    if not _is_step_aligned(value_ms, dt_ms):
        raise ConfigError(f"{name} must be an integer multiple of runtime.dt_ms")
    return int(round(value_ms / dt_ms))
