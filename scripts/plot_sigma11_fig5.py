from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


TARGET_TAU_S = 1.57
TARGET_TAU_T = 2.28
BLUE = "#0F4D92"
BLACK = "#272727"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def log_binned_distribution(values: np.ndarray, bins: int = 25) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return np.empty(0), np.empty(0)
    if values.min() == values.max():
        return np.asarray([values[0]]), np.asarray([1.0])
    edges = np.geomspace(values.min(), values.max() * (1.0 + 1e-9), bins + 1)
    hist, edges = np.histogram(values, bins=edges, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    keep = hist > 0
    return centers[keep], hist[keep]


def linear_fit_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    log_x = np.log10(x)
    log_y = np.log10(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    fitted = slope * log_x + intercept
    residual = np.sum((log_y - fitted) ** 2)
    total = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1.0 - residual / total if total > 0 else np.nan
    return float(slope), float(intercept), float(r_squared)


def average_size_given_duration(
    sizes: np.ndarray, durations: np.ndarray, min_count: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, counts = np.unique(durations, return_counts=True)
    keep = counts >= min_count
    x = unique[keep]
    y = np.asarray([sizes[durations == value].mean() for value in x])
    return x, y, counts[keep]


def analyze_run(run_dir: Path, sigma: float, seed: int) -> tuple[dict, dict]:
    source = run_dir / "analysis" / "avalanches_size_duration.csv"
    data = np.loadtxt(source, delimiter=",", skiprows=1, ndmin=2)
    sizes = data[:, 0]
    durations = data[:, 1]

    size_x, size_y = log_binned_distribution(sizes)
    duration_x, duration_y = log_binned_distribution(durations)
    if len(size_x) < 2 or len(duration_x) < 2:
        raise ValueError(f"Insufficient distribution bins in {source}")

    size_slope, size_intercept, r2_s = linear_fit_stats(size_x, size_y)
    duration_slope, duration_intercept, r2_t = linear_fit_stats(duration_x, duration_y)
    tau_s = -size_slope
    tau_t = -duration_slope

    st_x, st_y, st_counts = average_size_given_duration(sizes, durations)
    mask = (st_x > 0) & (st_y > 0)
    if mask.sum() < 2:
        raise ValueError(f"Insufficient size-duration bins in {source}")
    kappa_st, st_intercept, r2_st = linear_fit_stats(st_x[mask], st_y[mask])
    kappa_formula = (tau_t - 1.0) / (tau_s - 1.0)
    target_distance = np.sqrt(
        ((tau_s - TARGET_TAU_S) / TARGET_TAU_S) ** 2
        + ((tau_t - TARGET_TAU_T) / TARGET_TAU_T) ** 2
    )

    metrics = {
        "sigma": sigma,
        "seed": seed,
        "n_avalanches": int(len(sizes)),
        "tau_s": tau_s,
        "tau_t": tau_t,
        "r2_s": r2_s,
        "r2_t": r2_t,
        "kappa_st": kappa_st,
        "r2_st": r2_st,
        "kappa_formula": kappa_formula,
        "target_distance": float(target_distance),
        "run_dir": str(run_dir),
    }
    distribution = {
        "size_x": size_x,
        "size_y": size_y,
        "size_fit": 10 ** (size_intercept + size_slope * np.log10(size_x)),
        "duration_x": duration_x,
        "duration_y": duration_y,
        "duration_fit": 10 ** (duration_intercept + duration_slope * np.log10(duration_x)),
        "st_x": st_x[mask],
        "st_y": st_y[mask],
        "st_fit": 10 ** (st_intercept + kappa_st * np.log10(st_x[mask])),
        "st_counts": st_counts[mask],
    }
    return metrics, distribution


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold")


def plot_condition(metrics: dict, distribution: dict, output_stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    panels = (
        (
            "size_x", "size_y", "size_fit", r"Avalanche size $S$", r"$P(S)$",
            "Size distribution",
            rf"$\tau_S={metrics['tau_s']:.2f}$" + "\n" + rf"$R^2={metrics['r2_s']:.3f}$",
        ),
        (
            "duration_x", "duration_y", "duration_fit", r"Duration $T$ (ms)", r"$P(T)$",
            "Duration distribution",
            rf"$\tau_T={metrics['tau_t']:.2f}$" + "\n" + rf"$R^2={metrics['r2_t']:.3f}$",
        ),
        (
            "st_x", "st_y", "st_fit", r"Duration $T$ (ms)", r"$\langle S\rangle(T)$",
            "Size-duration scaling",
            rf"Direct fit: $\kappa_{{ST}}={metrics['kappa_st']:.2f}$"
            + "\n"
            + rf"From exponents: $\kappa_{{formula}}={metrics['kappa_formula']:.2f}$",
        ),
    )
    for label, ax, panel in zip("abc", axes, panels):
        x_key, y_key, fit_key, xlabel, ylabel, title, annotation = panel
        ax.loglog(
            distribution[x_key], distribution[y_key], "o-", ms=2.8, lw=1.0,
            color=BLUE, label="Binned data",
        )
        ax.loglog(
            distribution[x_key], distribution[fit_key], "--", lw=1.1,
            color=BLACK, label="Direct log-log fit",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=7.5, pad=5)
        ax.text(
            0.38 if label == "c" else 0.06,
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
        "Best BrainPy run: "
        + rf"$\sigma={metrics['sigma']:.2f}$, seed={metrics['seed']}, "
        + f"n={metrics['n_avalanches']}",
        fontsize=8,
    )
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_stem.with_suffix(f".{extension}"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def parse_run_name(run_dir: Path) -> tuple[float, int] | None:
    match = re.fullmatch(r"sigma_(\d+)p(\d+)_seed_(\d+)", run_dir.name)
    if not match:
        return None
    sigma = float(f"{match.group(1)}.{match.group(2)}")
    return sigma, int(match.group(3))


def discover_runs(scan_dir: Path, reused_run: Path | None) -> dict[tuple[float, int], Path]:
    runs: dict[tuple[float, int], Path] = {}
    for run_dir in scan_dir.glob("sigma_*_seed_*"):
        parsed = parse_run_name(run_dir)
        if parsed and (run_dir / "analysis" / "avalanches_size_duration.csv").exists():
            runs[parsed] = run_dir
    if reused_run and reused_run.exists():
        resolved = json.loads((reused_run / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((reused_run / "run_manifest.json").read_text(encoding="utf-8"))
        sigma = float(config.get("overrides", {}).get("runtime.sigma", 6.86))
        seed = int(resolved.get("seed", 1256874))
        runs[(sigma, seed)] = reused_run
    return runs


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", type=Path, default=Path("runs/fine_sigma_seed5"))
    parser.add_argument(
        "--reused-run", type=Path, default=Path("runs/full_sigma6p86_seed1256874")
    )
    parser.add_argument("--outdir", type=Path, default=Path("figures/brainpy_sigma11"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.scan_dir, args.reused_run)
    metrics_rows: list[dict] = []
    distributions: dict[tuple[float, int], dict] = {}
    for (sigma, seed), run_dir in sorted(runs.items()):
        metrics, distribution = analyze_run(run_dir, sigma, seed)
        metrics_rows.append(metrics)
        distributions[(sigma, seed)] = distribution

    sigma_values = sorted({row["sigma"] for row in metrics_rows})
    if len(sigma_values) != 11:
        raise RuntimeError(f"Expected 11 sigma groups, found {len(sigma_values)}: {sigma_values}")

    selected: list[dict] = []
    for sigma in sigma_values:
        group = [row for row in metrics_rows if row["sigma"] == sigma]
        if len(group) != 5:
            raise RuntimeError(f"Expected 5 seeds for sigma={sigma:.2f}, found {len(group)}")
        best = min(group, key=lambda row: row["target_distance"])
        selected.append(best)
        sigma_tag = f"{sigma:.2f}".replace(".", "p")
        plot_condition(
            best,
            distributions[(best["sigma"], best["seed"])],
            args.outdir / f"brainpy_best_sigma_{sigma_tag}",
        )

    write_csv(args.outdir / "all_55_run_metrics.csv", metrics_rows)
    write_csv(args.outdir / "selected_11_runs.csv", selected)
    manifest = {
        "selection_rule": "minimum normalized distance to tau_S=1.57 and tau_T=2.28 within each sigma",
        "number_of_sigma_groups": len(selected),
        "seeds_per_sigma": 5,
        "formats": ["png", "pdf", "svg"],
        "selected": selected,
    }
    (args.outdir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(selected)} three-panel figures in {args.outdir}")
    for row in selected:
        print(
            f"sigma={row['sigma']:.2f} seed={row['seed']} n={row['n_avalanches']} "
            f"tauS={row['tau_s']:.4f} tauT={row['tau_t']:.4f} "
            f"distance={row['target_distance']:.5f}"
        )


if __name__ == "__main__":
    main()
