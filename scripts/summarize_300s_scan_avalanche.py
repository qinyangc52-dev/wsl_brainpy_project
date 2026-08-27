#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import h5py
import numpy as np

from ecmm.analysis.avalanche import extract_avalanches


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from plot_sigma11_fig5 import analyze_run  # noqa: E402


PATTERN = re.compile(r"^sigma_(\d+)p(\d+)_seed_(\d+)$")
SUMMARY_FIELDS = (
    "n_avalanches", "tau_s", "tau_t", "kappa_formula", "kappa_st",
    "r2_s", "r2_t", "r2_st", "target_distance", "scaling_relative_error",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract comparable avalanche metrics for the 300 s scan.")
    parser.add_argument(
        "--scan-root", type=Path, default=PROJECT / "runs" / "fine_sigma_seed5_300s"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT / "comparison" / "brainpy_300s"
    )
    args = parser.parse_args()
    scan_root = args.scan_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    run_dirs = sorted(path for path in scan_root.iterdir() if path.is_dir() and PATTERN.match(path.name))
    if len(run_dirs) != 25:
        raise RuntimeError(f"Expected 25 runs, found {len(run_dirs)}")
    for index, run_dir in enumerate(run_dirs, start=1):
        match = PATTERN.match(run_dir.name)
        assert match is not None
        sigma = float(f"{match.group(1)}.{match.group(2)}")
        seed = int(match.group(3))
        analysis_dir = run_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        with h5py.File(run_dir / "run.h5", "r") as store:
            counts = np.asarray(store["rates/counts"])
            edges = np.asarray(store["rates/edges_ms"])
        sizes, durations = extract_avalanches(
            counts,
            source_bin_ms=float(edges[1] - edges[0]),
            avalanche_bin_ms=5.0,
        )
        np.savetxt(
            analysis_dir / "avalanches_size_duration.csv",
            np.column_stack((sizes, durations)),
            delimiter=",",
            header="size_spikes,duration_ms",
            comments="",
        )
        metrics, _ = analyze_run(run_dir, sigma, seed)
        metrics["scaling_relative_error"] = abs(
            metrics["kappa_formula"] - metrics["kappa_st"]
        ) / abs(metrics["kappa_formula"])
        rows.append(metrics)
        print(
            f"ANALYZED {index:02d}/25 sigma={sigma:.2f} seed={seed} "
            f"n={metrics['n_avalanches']}",
            flush=True,
        )

    write_csv(output_dir / "brainpy_300s_metrics.csv", rows)
    summary_rows: list[dict] = []
    for sigma in sorted({row["sigma"] for row in rows}):
        group = [row for row in rows if row["sigma"] == sigma]
        summary: dict[str, float | int] = {"sigma": sigma, "n_seeds": len(group)}
        for field in SUMMARY_FIELDS:
            values = np.asarray([row[field] for row in group], dtype=float)
            summary[f"{field}_mean"] = float(values.mean())
            summary[f"{field}_sd"] = float(values.std(ddof=1))
        summary_rows.append(summary)
    write_csv(output_dir / "brainpy_300s_summary.csv", summary_rows)
    print(f"SUMMARY COMPLETE output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
