from __future__ import annotations

import json
from pathlib import Path

from ..artifacts import resolve_run_artifact, validate_artifact
from ..config import load_config
from .avalanche import analyze_avalanches
from .legacy import export_legacy_outputs


def analyze_run(
    run_dir: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    make_figure: bool = True,
) -> dict:
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.resolved.yaml")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    artifact_dir = resolve_run_artifact(run_dir, config, manifest, artifact_dir)
    _, identity = validate_artifact(config, artifact_dir)
    if manifest.get("artifact_identity") not in (None, identity):
        raise ValueError("Run artifact identity mismatch")
    legacy_dir = export_legacy_outputs(run_dir, config, artifact_dir)
    avalanche = analyze_avalanches(run_dir, make_figure=make_figure)
    legacy_manifest = json.loads(
        (legacy_dir / "export_manifest.json").read_text(encoding="utf-8")
    )
    result = {
        "artifact_dir": str(artifact_dir),
        "legacy_dir": str(legacy_dir),
        "legacy_export": legacy_manifest,
        "avalanche": avalanche,
    }
    (run_dir / "analysis" / "analysis_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
