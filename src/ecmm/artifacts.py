from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse

from .config import ProjectConfig
from .connectome import file_sha256
from .patterns import PatternBank


ARTIFACT_FILES = (
    ("patterns.npz", "patterns_sha256"),
    ("connectivity_csr.npz", "connectivity_sha256"),
)


def artifact_rng_seeds(config: ProjectConfig) -> tuple[int, int]:
    """Return the legacy RNG seed pair used for offline artifact construction.

    The legacy program temporarily replaces the initialized network RNG with
    ``seed3`` when it is non-zero. In that case ``seed2`` is not used.
    """
    primary = config.seeds.effective_offline
    stream = 0 if config.seeds.offline else config.seeds.stream
    return primary, stream


def config_hash(config: ProjectConfig) -> str:
    encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def structural_hash(config: ProjectConfig) -> str:
    offline_seed, stream_seed = artifact_rng_seeds(config)
    payload = {
        "network": config.to_dict()["network"],
        # Keep the historic key so artifacts built with the default
        # offline=stream=0 configuration retain their existing identity.
        "network_seed": offline_seed,
        "dtype": config.artifact.dtype,
    }
    if stream_seed:
        payload["stream_seed"] = stream_seed
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_artifact(
    config: ProjectConfig,
    artifact_dir: str | Path,
) -> tuple[dict, dict[str, str]]:
    """Validate that an artifact is intact and belongs to this network config."""
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_structural_hash = structural_hash(config)
    actual_structural_hash = manifest.get("structural_hash")
    if actual_structural_hash != expected_structural_hash:
        raise ValueError(
            "Artifact structural hash mismatch: the configured network/seed/dtype "
            "does not match the offline artifact"
        )
    for filename, field in ARTIFACT_FILES:
        path = artifact_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact file: {path}")
        if file_sha256(path) != manifest.get(field):
            raise ValueError(f"Artifact integrity check failed for {filename}")
    identity = {
        key: str(manifest[key])
        for key in ("structural_hash", "patterns_sha256", "connectivity_sha256")
    }
    return manifest, identity


def portable_artifact_path(artifact_dir: str | Path, run_dir: str | Path) -> str | None:
    """Return a path which remains valid when the project tree is relocated."""
    try:
        return Path(os.path.relpath(Path(artifact_dir).resolve(), Path(run_dir).resolve())).as_posix()
    except ValueError:
        # Windows paths on different drives cannot be represented relatively.
        return None


def resolve_run_artifact(
    run_dir: str | Path,
    config: ProjectConfig,
    run_manifest: dict,
    override: str | Path | None = None,
) -> Path:
    """Resolve an artifact after a run directory or whole project has moved."""
    run_dir = Path(run_dir)
    if override is not None:
        candidate = Path(override).expanduser().resolve()
        if not (candidate / "manifest.json").exists():
            raise FileNotFoundError(f"Artifact override is invalid: {candidate}")
        return candidate

    candidates: list[Path] = []
    relative = run_manifest.get("artifact_relative")
    if relative:
        candidates.append((run_dir / relative).resolve())
    recorded = run_manifest.get("artifact_dir")
    if recorded:
        candidates.append(Path(recorded).expanduser())
    candidates.append((run_dir.resolve().parent.parent / "artifacts" / config.artifact.name))
    for candidate in candidates:
        if (candidate / "manifest.json").exists():
            return candidate.resolve()
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Unable to locate artifact for run {run_dir}. Checked: {checked}. "
        "Pass --artifact explicitly after relocating a standalone run directory."
    )


def save_artifact(
    output_dir: str | Path,
    config: ProjectConfig,
    tractography_path: str | Path,
    bank: PatternBank,
    weights: sparse.csr_matrix,
) -> Path:
    offline_seed, stream_seed = artifact_rng_seeds(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pattern_path = output / "patterns.npz"
    weight_path = output / "connectivity_csr.npz"
    np.savez_compressed(
        pattern_path,
        sites=bank.sites,
        who=bank.who,
        phi=bank.phi,
        H=bank.H,
        start=bank.start,
        where=bank.where,
        order=bank.order,
        posix=bank.posix,
        Z=bank.Z,
        K=bank.K,
    )
    sparse.save_npz(weight_path, weights, compressed=True)
    manifest = {
        "schema_version": 1,
        "artifact_name": config.artifact.name,
        "network_seed": offline_seed,
        "stream_seed": stream_seed,
        "shape": list(weights.shape),
        "nnz": int(weights.nnz),
        "dtype": str(weights.dtype),
        "matrix_orientation": "post_by_pre",
        "config_hash": config_hash(config),
        "structural_hash": structural_hash(config),
        "tractography_sha256": file_sha256(tractography_path),
        "patterns_sha256": file_sha256(pattern_path),
        "connectivity_sha256": file_sha256(weight_path),
        "structural_config": config.to_dict()["network"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
