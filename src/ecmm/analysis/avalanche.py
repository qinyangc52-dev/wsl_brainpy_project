from __future__ import annotations

import json
import warnings
from pathlib import Path

import h5py
import numpy as np


def extract_avalanches(
    module_counts: np.ndarray,
    source_bin_ms: float,
    avalanche_bin_ms: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    factor = int(round(avalanche_bin_ms / source_bin_ms))
    if factor <= 0 or not np.isclose(factor * source_bin_ms, avalanche_bin_ms):
        raise ValueError("avalanche_bin_ms must be an integer multiple of source bins")
    bins = len(module_counts) // factor
    activity = module_counts[:bins * factor].reshape(bins, factor, -1).sum(axis=(1, 2))
    sizes: list[float] = []
    durations: list[float] = []
    index = 0
    while index < len(activity):
        if activity[index] <= 0:
            index += 1
            continue
        start = index
        size = 0.0
        while index < len(activity) and activity[index] > 0:
            size += float(activity[index])
            index += 1
        sizes.append(size)
        durations.append((index - start) * avalanche_bin_ms)
    return np.asarray(sizes), np.asarray(durations)


def log_binned_distribution(values: np.ndarray, bins: int = 25):
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return np.empty(0), np.empty(0)
    if values.min() == values.max():
        return np.asarray([values[0]]), np.asarray([1.0])
    edges = np.geomspace(values.min(), values.max() * (1 + 1e-9), bins + 1)
    hist, edges = np.histogram(values, bins=edges, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    keep = hist > 0
    return centers[keep], hist[keep]


def simple_powerlaw_slope(values: np.ndarray) -> float | None:
    x, y = log_binned_distribution(values)
    if len(x) < 2:
        return None
    return float(-np.polyfit(np.log10(x), np.log10(y), 1)[0])


def analyze_avalanches(
    run_dir: str | Path,
    *,
    avalanche_bin_ms: float = 5.0,
    make_figure: bool = True,
) -> dict:
    run_dir = Path(run_dir)
    output = run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(run_dir / "run.h5", "r") as store:
        counts = np.asarray(store["rates/counts"])
        edges = np.asarray(store["rates/edges_ms"])
    source_bin_ms = float(edges[1] - edges[0])
    sizes, durations = extract_avalanches(counts, source_bin_ms, avalanche_bin_ms)
    np.savetxt(
        output / "avalanches_size_duration.csv",
        np.column_stack((sizes, durations)) if len(sizes) else np.empty((0, 2)),
        delimiter=",", header="size_spikes,duration_ms", comments="",
    )
    summary = {
        "avalanche_bin_ms": avalanche_bin_ms,
        "count": int(len(sizes)),
        "size_slope": simple_powerlaw_slope(sizes),
        "duration_slope": simple_powerlaw_slope(durations),
        "size_mean": float(sizes.mean()) if len(sizes) else 0.0,
        "duration_mean_ms": float(durations.mean()) if len(durations) else 0.0,
    }
    try:
        import powerlaw
        for label, values in (("size", sizes), ("duration", durations)):
            if len(values) >= 2:
                if len(np.unique(values)) < 3:
                    summary[f"{label}_formal"] = {
                        "status": "insufficient_data",
                        "reason": "fewer than three unique values",
                    }
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = powerlaw.Fit(values, discrete=True, verbose=False)
                        ratio, p_value = fit.distribution_compare("power_law", "exponential")
                    metrics = {
                        "alpha": float(fit.power_law.alpha), "xmin": float(fit.power_law.xmin),
                        "ks": float(fit.power_law.D), "loglikelihood_ratio": float(ratio),
                        "p_value": float(p_value),
                    }
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise ValueError("formal fit returned non-finite metrics")
                    summary[f"{label}_formal"] = metrics
                except (ValueError, FloatingPointError) as exc:
                    summary[f"{label}_formal"] = {
                        "status": "insufficient_data", "reason": str(exc)
                    }
    except ImportError:
        summary["formal_fit"] = "powerlaw package not installed"
    (output / "avalanche_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if make_figure and len(sizes):
        _plot_avalanche(sizes, durations, output)
    return summary


def _plot_avalanche(sizes: np.ndarray, durations: np.ndarray, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, values, label in zip(axes[:2], (sizes, durations), ("Size", "Duration (ms)")):
        x, y = log_binned_distribution(values)
        ax.loglog(x, y, "o-", color="black", markersize=3)
        ax.set_xlabel(label)
        ax.set_ylabel("Probability density")
    unique = np.unique(durations)
    mean_sizes = np.asarray([sizes[durations == value].mean() for value in unique])
    axes[2].loglog(unique, mean_sizes, "o-", color="black", markersize=3)
    axes[2].set_xlabel("Duration (ms)")
    axes[2].set_ylabel("Mean size")
    fig.tight_layout()
    fig.savefig(output / "avalanche_summary.png", dpi=300)
    fig.savefig(output / "avalanche_summary.pdf")
    plt.close(fig)
