from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from .legacy import legacy_scalar, parse_legacy_seed
from .schema import (
    ArtifactConfig,
    ConfigError,
    CueConfig,
    ExecutionConfig,
    IOConfig,
    MonitorConfig,
    NetworkConfig,
    ProjectConfig,
    RuntimeConfig,
    SeedConfig,
)


T = TypeVar("T")


LEGACY_MAP: dict[str, tuple[str, str]] = {
    "topo": ("network", "topology"), "S": ("network", "modules"),
    "Z": ("network", "neurons_per_module"), "G": ("network", "active_modules_per_pattern"),
    "K": ("network", "active_neurons_per_module"), "P": ("network", "patterns"),
    "sort": ("network", "sort"), "swap": ("network", "swap"),
    "range": ("network", "range"), "f": ("network", "frequency_hz"),
    "ddec": ("network", "ddec"), "dmax": ("network", "dmax"),
    "pmax": ("network", "pmax"),
    "sigma": ("runtime", "sigma"), "delta": ("runtime", "delta"),
    "alpha": ("runtime", "alpha"), "rho": ("runtime", "rho"),
    "bin": ("runtime", "output_bin_ms"), "tmax": ("runtime", "duration_ms"),
    "smin": ("runtime", "sigma_min"), "smax": ("runtime", "sigma_max"),
    "flush": ("monitors", "flush_bins"), "flush2": ("monitors", "overlap_history"),
    "tmin": ("monitors", "overlap_start_ms"), "twin": ("monitors", "overlap_window_ms"),
    "fmin": ("monitors", "overlap_min_fraction"), "pout": ("monitors", "patterns_observed"),
    "pout2": ("monitors", "playback_patterns"),
    "progress_interval": ("execution", "progress_interval_s"),
    "rstop": ("execution", "stop_rate_hz"), "nstop": ("execution", "stop_windows"),
    "tplay": ("execution", "playback_ms"), "cpu": ("execution", "cpu_limit_s"),
    "save": ("execution", "checkpoint_interval_s"),
    "mype": ("execution", "worker_index"), "debug": ("execution", "debug_level"),
    "seed": ("seeds", "network"), "seed2": ("seeds", "stream"),
    "seed3": ("seeds", "offline"), "seed4": ("seeds", "dynamics"),
    "name": ("io", "run_name"), "tmpdir": ("io", "temp_dir"),
    "outdir": ("io", "output_dir"), "file": ("io", "legacy_file_mode"),
    "maxsp": ("io", "max_spikes"), "stdout": ("io", "stdout_path"),
}


def load_config(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
    *,
    strict_legacy: bool = False,
) -> ProjectConfig:
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config = config_from_dict(payload)
    else:
        config = load_legacy_config(path, strict=strict_legacy)
    if overrides:
        config = apply_overrides(config, overrides)
    return config.validate()


def load_legacy_config(path: str | Path, *, strict: bool = False) -> ProjectConfig:
    parsed = parse_legacy_seed(path)
    sections: dict[str, dict[str, Any]] = {}
    unmapped: dict[str, str] = {}
    for key, raw in parsed.values.items():
        if key == "noise":
            sections.setdefault("runtime", {})["noise_mode"] = (
                "constant" if int(legacy_scalar(raw)) else "gaussian"
            )
            continue
        target = LEGACY_MAP.get(key)
        if target is None:
            unmapped[key] = raw
            continue
        section, field_name = target
        sections.setdefault(section, {})[field_name] = legacy_scalar(raw)
    if strict and unmapped:
        raise ConfigError("Unknown legacy SEED keys: " + ", ".join(sorted(unmapped)))
    sections["cues"] = [cue.__dict__ for cue in parsed.cues]
    sections["legacy_unmapped"] = unmapped
    return config_from_dict(sections)


def config_from_dict(payload: Mapping[str, Any]) -> ProjectConfig:
    allowed = {field.name for field in fields(ProjectConfig)}
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigError("Unknown top-level config keys: " + ", ".join(sorted(unknown)))
    return ProjectConfig(
        network=_construct(NetworkConfig, payload.get("network", {}), "network"),
        runtime=_construct(RuntimeConfig, payload.get("runtime", {}), "runtime"),
        seeds=_construct(SeedConfig, payload.get("seeds", {}), "seeds"),
        artifact=_construct(ArtifactConfig, payload.get("artifact", {}), "artifact"),
        monitors=_construct(MonitorConfig, payload.get("monitors", {}), "monitors"),
        execution=_construct(ExecutionConfig, payload.get("execution", {}), "execution"),
        io=_construct(IOConfig, payload.get("io", {}), "io"),
        cues=tuple(_construct(CueConfig, cue, "cues[]") for cue in payload.get("cues", ())),
        legacy_unmapped=dict(payload.get("legacy_unmapped", {})),
    )


def _construct(cls: type[T], values: Mapping[str, Any], section: str) -> T:
    if not isinstance(values, Mapping):
        raise ConfigError(f"{section} must be a mapping")
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError(f"Unknown {section} keys: " + ", ".join(sorted(unknown)))
    normalized = dict(values)
    for key in ("neurons_per_module", "active_neurons_per_module"):
        if key in normalized and isinstance(normalized[key], list):
            normalized[key] = tuple(normalized[key])
    return cls(**normalized)


def apply_overrides(config: ProjectConfig, overrides: Mapping[str, Any]) -> ProjectConfig:
    payload = config.to_dict()
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        if len(parts) != 2 or parts[0] not in payload or not isinstance(payload[parts[0]], dict):
            raise ConfigError(f"Override must name a section and field: {dotted_key}")
        if parts[1] not in payload[parts[0]]:
            raise ConfigError(f"Unknown override: {dotted_key}")
        payload[parts[0]][parts[1]] = value
    return config_from_dict(payload).validate()


def parse_override_list(items: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"Override must use section.field=value: {item}")
        key, raw = item.split("=", 1)
        result[key] = yaml.safe_load(raw)
    return result


def dump_config(config: ProjectConfig, path: str | Path | None = None) -> str:
    payload = config.to_dict()
    if not payload["legacy_unmapped"]:
        payload.pop("legacy_unmapped")
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
