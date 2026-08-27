#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from plot_sigma11_fig5 import analyze_run  # noqa: E402


RUN_PATTERN = re.compile(r"^sigma_(\d+)p(\d+)_seed_(\d+)$")
BLUE = "#0F4D92"
BLACK = "#272727"


def parse_run(run_dir: Path) -> tuple[float, int]:
    match = RUN_PATTERN.fullmatch(run_dir.name)
    if match is None:
        raise ValueError(f"Unrecognized run directory: {run_dir}")
    sigma = float(f"{match.group(1)}.{match.group(2)}")
    return sigma, int(match.group(3))


def padded_log_limits(values: list[np.ndarray], pad_decades: float = 0.08) -> tuple[float, float]:
    positive = np.concatenate([np.asarray(value)[np.asarray(value) > 0] for value in values])
    low = 10 ** (np.log10(positive.min()) - pad_decades)
    high = 10 ** (np.log10(positive.max()) + pad_decades)
    return float(low), float(high)


def global_panel_limits(distributions: list[dict]) -> dict[str, tuple[float, float]]:
    return {
        "size_x": padded_log_limits([item["size_x"] for item in distributions]),
        "size_y": padded_log_limits([item["size_y"] for item in distributions]),
        "duration_x": padded_log_limits([item["duration_x"] for item in distributions]),
        "duration_y": padded_log_limits([item["duration_y"] for item in distributions]),
        "st_x": padded_log_limits([item["st_x"] for item in distributions]),
        "st_y": padded_log_limits([item["st_y"] for item in distributions]),
    }


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_run(
    metrics: dict,
    distribution: dict,
    limits: dict[str, tuple[float, float]],
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    panels = (
        (
            "size_x",
            "size_y",
            "size_fit",
            r"Avalanche size $S$",
            r"$P(S)$",
            "Size distribution",
            rf"$\tau_S={metrics['tau_s']:.2f}$" + "\n" + rf"$R^2={metrics['r2_s']:.3f}$",
        ),
        (
            "duration_x",
            "duration_y",
            "duration_fit",
            r"Duration $T$ (ms)",
            r"$P(T)$",
            "Duration distribution",
            rf"$\tau_T={metrics['tau_t']:.2f}$" + "\n" + rf"$R^2={metrics['r2_t']:.3f}$",
        ),
        (
            "st_x",
            "st_y",
            "st_fit",
            r"Duration $T$ (ms)",
            r"$\langle S\rangle(T)$",
            "Size-duration scaling",
            rf"Direct fit: $\kappa_{{ST}}={metrics['kappa_st']:.2f}$"
            + "\n"
            + rf"From exponents: $\kappa_{{formula}}={metrics['kappa_formula']:.2f}$",
        ),
    )

    for label, ax, panel in zip("abc", axes, panels):
        x_key, y_key, fit_key, xlabel, ylabel, title, annotation = panel
        ax.loglog(
            distribution[x_key],
            distribution[y_key],
            "o-",
            ms=2.8,
            lw=1.0,
            color=BLUE,
            label="Binned data",
        )
        ax.loglog(
            distribution[x_key],
            distribution[fit_key],
            "--",
            lw=1.1,
            color=BLACK,
            label="Direct log-log fit",
        )
        ax.set_xlim(limits[x_key])
        ax.set_ylim(limits[y_key])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=7.5, pad=5)
        ax.text(
            0.39 if label == "c" else 0.06,
            0.08,
            annotation,
            transform=ax.transAxes,
            fontsize=6.5,
            va="bottom",
        )
        if label == "c":
            ax.legend(loc="upper left", fontsize=5.8)
        add_panel_label(ax, label)
        ax.tick_params(direction="out", length=3, width=0.7)

    fig.suptitle(
        "BrainPy 300 s run: "
        + rf"$\sigma={metrics['sigma']:.2f}$, seed={metrics['seed']}, "
        + f"n={metrics['n_avalanches']}",
        fontsize=8,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_metrics(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot all 25 BrainPy 300 s avalanche runs.")
    parser.add_argument(
        "--scan-dir", type=Path, default=PROJECT / "runs" / "fine_sigma_seed5_300s"
    )
    parser.add_argument(
        "--outdir", type=Path, default=PROJECT / "figures" / "brainpy_300s_all_25"
    )
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    run_dirs = sorted(
        path
        for path in args.scan_dir.iterdir()
        if path.is_dir()
        and RUN_PATTERN.fullmatch(path.name)
        and (path / "analysis" / "avalanches_size_duration.csv").exists()
    )
    if len(run_dirs) != 25:
        raise RuntimeError(f"Expected 25 analyzed runs, found {len(run_dirs)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict] = []
    distributions: list[dict] = []
    for run_dir in run_dirs:
        sigma, seed = parse_run(run_dir)
        metrics, distribution = analyze_run(run_dir, sigma, seed)
        metrics_rows.append(metrics)
        distributions.append(distribution)

    limits = global_panel_limits(distributions)
    image_paths: list[Path] = []
    for metrics, distribution in zip(metrics_rows, distributions):
        sigma_tag = f"{metrics['sigma']:.2f}".replace(".", "p")
        output_path = args.outdir / (
            f"brainpy_300s_sigma_{sigma_tag}_seed_{metrics['seed']}.png"
        )
        plot_run(metrics, distribution, limits, output_path, args.dpi)
        image_paths.append(output_path)
        print(f"WROTE {output_path.name}", flush=True)

    write_metrics(args.outdir / "figure_metrics.csv", metrics_rows)
    manifest = {
        "schema_version": 1,
        "figure_contract": {
            "core_conclusion": (
                "Diagnose run-level avalanche size, duration, and size-duration scaling "
                "across the 25 BrainPy 300 s simulations."
            ),
            "archetype": "quantitative triptych",
            "reviewer_risk": (
                "Log-binned linear fits are diagnostic and do not alone establish a "
                "statistically validated power law or criticality."
            ),
        },
        "backend": "Python/matplotlib",
        "formats": ["png"],
        "dpi": args.dpi,
        "runs": len(image_paths),
        "shared_axis_limits": limits,
        "images": [path.name for path in image_paths],
    }
    (args.outdir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"COMPLETE figures={len(image_paths)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
