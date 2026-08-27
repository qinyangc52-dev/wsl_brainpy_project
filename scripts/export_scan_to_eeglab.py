#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np

from ecmm.eeglab import export_run_to_eeglab


PROJECT = Path(__file__).resolve().parents[1]
SIGMA_TAGS = ("6p85", "6p86", "6p87", "6p90", "6p95")
SEEDS = tuple(range(1256874, 1256879))
RUN_PATTERN = re.compile(r"^sigma_(6p\d+)_seed_(\d+)$")


def expected_runs(scan_root: Path) -> list[tuple[str, int, Path]]:
    expected = [
        (sigma_tag, seed, scan_root / f"sigma_{sigma_tag}_seed_{seed}")
        for sigma_tag in SIGMA_TAGS
        for seed in SEEDS
    ]
    actual = {path.name for path in scan_root.iterdir() if path.is_dir() and RUN_PATTERN.match(path.name)}
    expected_names = {path.name for _, _, path in expected}
    missing = sorted(expected_names - actual)
    extra = sorted(actual - expected_names)
    if missing or extra:
        raise RuntimeError(f"Scan run set mismatch; missing={missing}, extra={extra}")
    return expected


def validate_run(sigma_tag: str, seed: int, run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    run_h5 = run_dir / "run.h5"
    if not summary_path.is_file() or not run_h5.is_file():
        raise FileNotFoundError(f"Missing summary.json or run.h5: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or float(summary.get("completed_ms", 0.0)) < 300000.0:
        raise RuntimeError(f"Run is not a complete 300 s simulation: {run_dir}")

    with h5py.File(run_h5, "r") as store:
        if "rates/module_hz" not in store or "rates/edges_ms" not in store:
            raise KeyError(f"Missing rates datasets: {run_h5}")
        signal = store["rates/module_hz"]
        edges = np.asarray(store["rates/edges_ms"], dtype=np.float64).reshape(-1)
        if signal.shape != (300000, 66):
            raise RuntimeError(f"Unexpected signal shape {signal.shape}: {run_h5}")
        if edges.shape != (300001,):
            raise RuntimeError(f"Unexpected edge shape {edges.shape}: {run_h5}")
        steps = np.diff(edges)
        if not np.all(np.isfinite(steps)) or not np.allclose(steps, 1.0, rtol=1e-6, atol=1e-9):
            raise RuntimeError(f"Signal is not uniformly sampled at 1000 Hz: {run_h5}")

        total = 0
        total_sum = 0.0
        total_sq = 0.0
        minimum = np.inf
        maximum = -np.inf
        for start in range(0, signal.shape[0], 10000):
            block = np.asarray(signal[start : start + 10000], dtype=np.float64)
            if not np.all(np.isfinite(block)):
                raise RuntimeError(f"NaN or infinite values found: {run_h5}")
            total += block.size
            total_sum += float(block.sum(dtype=np.float64))
            total_sq += float(np.square(block).sum(dtype=np.float64))
            minimum = min(minimum, float(block.min()))
            maximum = max(maximum, float(block.max()))
    mean = total_sum / total
    variance = max(total_sq / total - mean * mean, 0.0)
    return {
        "sigma": sigma_tag.replace("p", "."),
        "sigma_tag": sigma_tag,
        "seed": seed,
        "subject": seed - SEEDS[0] + 1,
        "source_run": str(run_dir.resolve()),
        "source_samples": 300000,
        "channels": 66,
        "input_sampling_hz": 1000.0,
        "duration_seconds": 300.0,
        "finite": True,
        "source_min_hz": minimum,
        "source_max_hz": maximum,
        "source_mean_hz": mean,
        "source_std_hz": variance**0.5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and export the 25-run 300 s scan to EEGLAB.")
    parser.add_argument(
        "--scan-root", type=Path, default=PROJECT / "runs" / "fine_sigma_seed5_300s"
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT / "eeglab_exports" / "fine_sigma_seed5_300s_500Hz",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scan_root = args.scan_root.resolve()
    output_root = args.output_root.resolve()
    if not scan_root.is_dir():
        raise FileNotFoundError(f"Scan root not found: {scan_root}")

    runs = expected_runs(scan_root)
    qc_rows = []
    for index, (sigma_tag, seed, run_dir) in enumerate(runs, start=1):
        row = validate_run(sigma_tag, seed, run_dir)
        qc_rows.append(row)
        print(f"VALID {index:02d}/25 sigma={row['sigma']} seed={seed}", flush=True)
    print("SOURCE VALIDATION COMPLETE 25/25", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(qc_rows, start=1):
        sigma_tag = row["sigma_tag"]
        seed = int(row["seed"])
        subject = int(row["subject"])
        run_dir = scan_root / f"sigma_{sigma_tag}_seed_{seed}"
        condition = f"sigma_{sigma_tag}"
        output_dir = output_root / condition
        stem = f"sub-{subject:02d}_task-ecmm"
        exported = export_run_to_eeglab(
            run_dir,
            output_dir,
            filename_stem=stem,
            target_sfreq_hz=500.0,
            subject=f"{subject:02d}",
            condition=condition,
            overwrite=args.overwrite,
        )
        row.update({
            "output_sampling_hz": exported["output_sampling_hz"],
            "output_samples": exported["samples"],
            "set_file": str((output_dir / exported["set_file"]).resolve()),
            "fdt_file": str((output_dir / exported["fdt_file"]).resolve()),
            "set_sha256": exported["sha256"]["set"],
            "fdt_sha256": exported["sha256"]["fdt"],
        })
        print(f"EXPORTED {index:02d}/25 {condition} {stem}", flush=True)

    fieldnames = list(qc_rows[0])
    qc_path = output_root / "batch_export_qc.csv"
    with qc_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(qc_rows)
    manifest = {
        "schema_version": 1,
        "source_scan_root": str(scan_root),
        "output_root": str(output_root),
        "signal_definition": "Regional population firing rate in Hz; not scalp EEG voltage",
        "conditions": [f"sigma_{tag}" for tag in SIGMA_TAGS],
        "seeds": list(SEEDS),
        "subjects": {str(seed): seed - SEEDS[0] + 1 for seed in SEEDS},
        "runs": 25,
        "channels": 66,
        "source_sampling_hz": 1000.0,
        "output_sampling_hz": 500.0,
        "source_samples": 300000,
        "output_samples": 150000,
        "duration_seconds": 300.0,
        "resampling": "scipy.signal.resample_poly with anti-aliasing",
        "qc_csv": qc_path.name,
    }
    manifest_path = output_root / "batch_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"BATCH EXPORT COMPLETE 25/25 output={output_root}", flush=True)
    print(f"QC={qc_path}", flush=True)
    print(f"MANIFEST={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
