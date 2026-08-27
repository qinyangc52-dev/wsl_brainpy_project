from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import brainpy as bp
import brainpy.math as bm
import jax
import numpy as np
from scipy import sparse

from ..artifacts import config_hash, portable_artifact_path, validate_artifact
from ..config import ProjectConfig, dump_config
from ..models import CueInput, ECMMBrainPyNetwork, NoiseInput, SigmaScheduler
from ..monitors import MonitorSuite
from .checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from .store import RunStore


@dataclass(frozen=True)
class RunResult:
    output_dir: Path
    status: str
    completed_step: int
    total_spikes: int
    wall_seconds: float


def gpu_memory_mib() -> dict[str, int] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        used, total = (int(value.strip()) for value in result.stdout.splitlines()[0].split(","))
        return {"used": used, "total": total}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class SimulationRunner:
    def __init__(self, config: ProjectConfig, artifact_dir: str | Path, output_dir: str | Path):
        self.config = config.validate()
        self.artifact_dir = Path(artifact_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_path = self.output_dir / "checkpoint.npz"
        self.store_path = self.output_dir / "run.h5"
        self.summary_path = self.output_dir / "summary.json"
        self.manifest_path = self.artifact_dir / "manifest.json"
        self.artifact_manifest, self.identity = validate_artifact(
            self.config, self.artifact_dir
        )
        self.run_config_hash = config_hash(self.config)

    def run(self, *, resume: bool = False, max_chunks: int | None = None) -> RunResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not resume and (self.store_path.exists() or self.checkpoint_path.exists()):
            raise FileExistsError(
                f"Run output already exists: {self.output_dir}. Use resume or a new directory."
            )
        patterns = np.load(self.artifact_dir / "patterns.npz", allow_pickle=False)
        weights = sparse.load_npz(self.artifact_dir / "connectivity_csr.npz").astype(np.float32)
        model = self._build_model(weights)
        noise = NoiseInput.from_config(self.config.seeds.effective_dynamics, self.config.runtime)
        cue = CueInput.from_patterns(
            self.config.cues,
            patterns["who"],
            patterns["H"],
            self.config.runtime.dt_ms,
            weights.shape[0],
        )
        sigma = SigmaScheduler.from_config(self.config.runtime)
        monitors = MonitorSuite(self.config, patterns)
        completed_step = 0
        total_spikes = 0
        stop_windows = 0
        stop_windows_checked = 0
        create_store = not resume
        if resume:
            checkpoint = self._validated_checkpoint()
            completed_step = checkpoint.step
            total_spikes = checkpoint.total_spikes
            model.load_state_dict(checkpoint.network_state)
            monitors.restore(checkpoint.monitor_state)
            stop_windows = int(checkpoint.monitor_state.get("runner_stop_windows", np.asarray(0)))
            stop_windows_checked = int(
                checkpoint.monitor_state.get("runner_stop_windows_checked", np.asarray(0))
            )
        metadata = {
            "config_hash": self.run_config_hash,
            "artifact_identity": self.identity,
            "created_unix": time.time(),
        }
        store = RunStore(self.store_path, create=create_store, metadata=metadata)
        if resume:
            if store.metadata.get("config_hash") != self.run_config_hash:
                store.close()
                raise ValueError("Run store configuration hash does not match resume configuration")
            store.truncate_spikes(checkpoint.recorded_spikes)
        else:
            initial_monitor_state = monitors.snapshot()
            initial_monitor_state["runner_stop_windows"] = np.asarray(0)
            initial_monitor_state["runner_stop_windows_checked"] = np.asarray(0)
            self._save_checkpoint(
                model, initial_monitor_state, 0, store.spike_count, 0, complete=False
            )
        dump_config(self.config, self.output_dir / "config.resolved.yaml")
        self._write_run_manifest(status="running", resumed=resume)
        total_steps = int(round(self.config.runtime.duration_ms / self.config.runtime.dt_ms))
        chunk_steps = int(round(self.config.runtime.chunk_ms / self.config.runtime.dt_ms))
        started = time.perf_counter()
        chunk_timings: list[float] = []
        status = "complete"
        chunks_run = 0
        next_progress_s = self.config.execution.progress_interval_s
        checkpoint_interval_s = self.config.execution.checkpoint_interval_s
        next_checkpoint_s = checkpoint_interval_s
        caught_error: Exception | None = None
        error_detail: dict[str, str] | None = None
        memory_before = gpu_memory_mib()
        try:
            while completed_step < total_steps:
                if max_chunks is not None and chunks_run >= max_chunks:
                    status = "paused"
                    break
                steps = min(chunk_steps, total_steps - completed_step)
                if (self.config.execution.stop_rate_hz > 0 and
                        self.config.execution.stop_windows > 0):
                    flush_steps = int(round(
                        self.config.monitors.flush_bins *
                        self.config.runtime.output_bin_ms / self.config.runtime.dt_ms
                    ))
                    next_flush_step = (completed_step // flush_steps + 1) * flush_steps
                    steps = min(steps, next_flush_step - completed_step)
                noise_values = noise.chunk(completed_step, steps, model.num)
                cue_values = bm.asarray(cue.chunk(completed_step, steps))
                sigma_values = sigma.values(completed_step, steps)
                chunk_start = time.perf_counter()
                spike_matrix, cue_matrix = bm.for_loop(
                    model.step_with_metadata,
                    (noise_values, cue_values, sigma_values),
                    jit=True,
                )
                spike_matrix, cue_matrix = (
                    np.asarray(jax.device_get(spike_matrix)),
                    np.asarray(jax.device_get(cue_matrix)),
                )
                chunk_timings.append(time.perf_counter() - chunk_start)
                local_step, neurons = np.nonzero(spike_matrix)
                times = (completed_step + local_step + 1).astype(np.float32) * self.config.runtime.dt_ms
                cue_flags = cue_matrix[local_step, neurons]
                store.append_spikes(times, neurons.astype(np.int32), cue_flags)
                total_spikes += len(times)
                completed_step += steps
                chunks_run += 1
                monitors.update_chunk(times, neurons.astype(np.int32), completed_step * self.config.runtime.dt_ms)
                if not self._state_is_finite(model):
                    status = "non_finite_state"
                if self.config.execution.stop_rate_hz > 0:
                    completed_bins = int(
                        completed_step * self.config.runtime.dt_ms /
                        self.config.runtime.output_bin_ms + 1e-9
                    )
                    available_windows = completed_bins // self.config.monitors.flush_bins
                    while stop_windows_checked < available_windows:
                        stop_windows_checked += 1
                        window_rate = monitors.flush_window_rate_hz(stop_windows_checked)
                        stop_windows = (
                            stop_windows + 1
                            if window_rate > self.config.execution.stop_rate_hz
                            else 0
                        )
                        if (self.config.execution.stop_windows > 0 and
                                stop_windows >= self.config.execution.stop_windows):
                            status = "rate_stop"
                            break
                monitor_state = monitors.snapshot()
                monitor_state["runner_stop_windows"] = np.asarray(stop_windows)
                monitor_state["runner_stop_windows_checked"] = np.asarray(stop_windows_checked)
                elapsed = time.perf_counter() - started
                if checkpoint_interval_s == 0 or elapsed >= next_checkpoint_s:
                    self._save_checkpoint(
                        model,
                        monitor_state,
                        completed_step,
                        store.spike_count,
                        total_spikes,
                        complete=False,
                    )
                    if checkpoint_interval_s > 0:
                        while next_checkpoint_s <= elapsed:
                            next_checkpoint_s += checkpoint_interval_s
                if (self.config.execution.progress_interval_s > 0 and
                        (chunks_run == 1 or elapsed >= next_progress_s or completed_step >= total_steps)):
                    self._print_progress(completed_step, total_steps, total_spikes, started)
                    while next_progress_s <= elapsed:
                        next_progress_s += self.config.execution.progress_interval_s
                if status != "complete":
                    break
                if self.config.execution.cpu_limit_s > 0 and time.perf_counter() - started >= self.config.execution.cpu_limit_s:
                    status = "wall_time_limit"
                    break
            if status == "paused":
                monitor_state = monitors.snapshot()
                monitor_state["runner_stop_windows"] = np.asarray(stop_windows)
                monitor_state["runner_stop_windows_checked"] = np.asarray(stop_windows_checked)
                self._save_checkpoint(
                    model, monitor_state, completed_step, store.spike_count, total_spikes, complete=False
                )
        except KeyboardInterrupt:
            status = "interrupted"
            monitor_state = monitors.snapshot()
            monitor_state["runner_stop_windows"] = np.asarray(stop_windows)
            monitor_state["runner_stop_windows_checked"] = np.asarray(stop_windows_checked)
            self._save_checkpoint(
                model, monitor_state, completed_step, store.spike_count, total_spikes, complete=False
            )
        except Exception as exc:
            status = "error"
            caught_error = exc
            error_detail = {"type": type(exc).__name__, "message": str(exc)}
            monitor_state = monitors.snapshot()
            monitor_state["runner_stop_windows"] = np.asarray(stop_windows)
            monitor_state["runner_stop_windows_checked"] = np.asarray(stop_windows_checked)
            self._save_checkpoint(
                model, monitor_state, completed_step, store.spike_count, total_spikes, complete=False
            )
        wall_seconds = time.perf_counter() - started
        finalized = None
        if status not in ("paused", "interrupted", "error") and completed_step > 0:
            try:
                finalized = monitors.finalize(
                    store, completed_ms=completed_step * self.config.runtime.dt_ms
                )
                monitor_state = monitors.snapshot()
                monitor_state["runner_stop_windows"] = np.asarray(stop_windows)
                monitor_state["runner_stop_windows_checked"] = np.asarray(stop_windows_checked)
                self._save_checkpoint(
                    model,
                    monitor_state,
                    completed_step,
                    store.spike_count,
                    total_spikes,
                    complete=completed_step >= total_steps,
                )
            except Exception as exc:
                status = "error"
                caught_error = exc
                error_detail = {"type": type(exc).__name__, "message": str(exc)}
        summary = {
            "schema_version": 2,
            "status": status,
            "backend": "BrainPy/JAX",
            "brainpy_version": bp.__version__,
            "jax_version": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "platform": platform.platform(),
            "config_hash": self.run_config_hash,
            "artifact_identity": self.identity,
            "completed_step": completed_step,
            "completed_ms": completed_step * self.config.runtime.dt_ms,
            "duration_ms": self.config.runtime.duration_ms,
            "total_spikes": total_spikes,
            "mean_rate_hz": 1000.0 * total_spikes / max(
                1.0, self.config.network.total_neurons * completed_step * self.config.runtime.dt_ms
            ),
            "wall_seconds": wall_seconds,
            "chunk_timings_s": chunk_timings,
            "gpu_memory_before_mib": memory_before,
            "gpu_memory_after_mib": gpu_memory_mib(),
            "resumed": resume,
            "finalized": finalized is not None,
            "error": error_detail,
        }
        _atomic_json(self.summary_path, summary)
        self._write_run_manifest(status=status, resumed=resume)
        store.close()
        if caught_error is not None:
            raise RuntimeError(
                f"Simulation failed after step {completed_step}; checkpoint saved at "
                f"{self.checkpoint_path}"
            ) from caught_error
        return RunResult(self.output_dir, status, completed_step, total_spikes, wall_seconds)

    def _build_model(self, weights):
        runtime = self.config.runtime
        return ECMMBrainPyNetwork(
            weights,
            dt_ms=runtime.dt_ms,
            sigma=runtime.sigma,
            delta=runtime.delta,
            tau_a_ms=runtime.tau_a_ms,
            tau_b_ms=runtime.tau_b_ms,
            threshold=runtime.threshold,
        )

    def _validated_checkpoint(self) -> Checkpoint:
        if not self.checkpoint_path.exists() or not self.store_path.exists():
            raise FileNotFoundError("Resume requires checkpoint.npz and run.h5")
        checkpoint = load_checkpoint(self.checkpoint_path)
        if checkpoint.config_hash != self.run_config_hash:
            raise ValueError("Checkpoint configuration hash mismatch")
        if checkpoint.artifact_identity != self.identity:
            raise ValueError("Checkpoint artifact identity mismatch")
        return checkpoint

    def _save_checkpoint(self, model, monitor_state, step, recorded, total, *, complete):
        save_checkpoint(
            self.checkpoint_path,
            Checkpoint(
                step=step,
                recorded_spikes=recorded,
                total_spikes=total,
                config_hash=self.run_config_hash,
                artifact_identity=self.identity,
                network_state=model.state_dict(),
                monitor_state=monitor_state,
                complete=complete,
            ),
        )

    @staticmethod
    def _state_is_finite(model) -> bool:
        state = model.state_dict()
        return bool(np.isfinite(state["A"]).all() and np.isfinite(state["B"]).all())

    def _print_progress(self, step, total_steps, spikes, started):
        elapsed = time.perf_counter() - started
        progress = step / total_steps
        eta = elapsed * (1.0 - progress) / progress if progress > 0 else float("inf")
        print(
            f"PROGRESS {100*progress:6.2f}% sim={step*self.config.runtime.dt_ms:12g} ms "
            f"elapsed={elapsed:9.1f} s eta={eta:9.1f} s spikes={spikes:12d}",
            flush=True,
        )

    def _write_run_manifest(self, *, status: str, resumed: bool):
        _atomic_json(
            self.output_dir / "run_manifest.json",
            {
                "schema_version": 2,
                "status": status,
                "resumed": resumed,
                "config_hash": self.run_config_hash,
                "artifact_identity": self.identity,
                "artifact_dir": str(self.artifact_dir.resolve()),
                "artifact_relative": portable_artifact_path(
                    self.artifact_dir, self.output_dir
                ),
                "updated_unix": time.time(),
            },
        )


def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)
