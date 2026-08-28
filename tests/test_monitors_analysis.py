from dataclasses import replace
import json
from pathlib import Path

import h5py
import numpy as np

from ecmm.analysis import analyze_run, extract_avalanches
from ecmm.config import CueConfig, load_config
from ecmm.monitors import PatternOverlapMonitor, activity_statistics
from ecmm.runtime import SimulationRunner


PROJECT = Path(__file__).resolve().parents[1]
ARTIFACT = PROJECT / "artifacts" / "prototype_seed_1256878"


def analysis_config():
    base = load_config(PROJECT / "configs" / "prototype.yaml")
    return replace(
        base,
        runtime=replace(base.runtime, duration_ms=20.0, chunk_ms=10.0),
        monitors=replace(
            base.monitors,
            flush_bins=5,
            overlap_start_ms=0.0,
            overlap_window_ms=10.0,
            overlap_min_fraction=0.1,
        ),
        execution=replace(base.execution, progress_interval_s=0.0, playback_ms=10.0),
        io=replace(base.io, run_name="analysis-test"),
        cues=(CueConfig(pattern=0, start_ms=1.0, spike_count=5, frequency_hz=100.0),),
    )


def test_activity_statistics_matches_manual_counts():
    config = analysis_config()
    counts = np.zeros((20, config.network.modules), dtype=np.int64)
    counts[:5, 0] = [0, 1, 2, 1, 1]
    window, cumulative = activity_statistics(config, counts)
    assert window.shape == (4, 8)
    np.testing.assert_allclose(window[0, 5], np.var([0, 1, 2, 1, 1]))
    np.testing.assert_allclose(window[0, 7], np.var([0, 1, 2, 1, 1]) / 1.0)
    np.testing.assert_allclose(cumulative[0], window[0])


def test_avalanche_extraction_uses_silent_bins_as_boundaries():
    counts = np.array([[1], [1], [0], [2], [0], [0]], dtype=np.int64)
    sizes, durations = extract_avalanches(counts, source_bin_ms=1.0, avalanche_bin_ms=1.0)
    np.testing.assert_array_equal(sizes, [2, 2])
    np.testing.assert_array_equal(durations, [2, 1])


def test_overlap_windows_advance_through_silent_periods():
    config = analysis_config()
    config = replace(
        config,
        monitors=replace(
            config.monitors,
            flush_bins=10,
            overlap_start_ms=0.0,
            overlap_window_ms=5.0,
            overlap_min_fraction=0.5,
        ),
    )
    patterns = {
        "order": np.array([[0]], dtype=np.int32),
        "phi": np.array([[0.0]], dtype=np.float64),
        "H": np.array([1], dtype=np.int32),
    }
    monitor = PatternOverlapMonitor(config, patterns)
    monitor.update(
        np.array([6.0, 7.0, 8.0]),
        np.array([0, 0, 0], dtype=np.int32),
        chunk_end_ms=10.0,
    )
    monitor.update(
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.int32),
        chunk_end_ms=20.0,
    )

    assert [row["window_end_ms"] for row in monitor.rows] == [10.0, 20.0]
    assert monitor.rows[0]["spikes"] == 3
    assert monitor.rows[0]["max_overlap"] > 0.0
    assert monitor.rows[1]["window_start_ms"] == 15.0
    assert monitor.rows[1]["spikes"] == 0
    assert monitor.rows[1]["max_overlap"] == 0.0


def test_end_to_end_monitors_legacy_and_avalanche(tmp_path):
    output = tmp_path / "run"
    config = analysis_config()
    config = replace(config, io=replace(config.io, max_spikes=2))
    result = SimulationRunner(config, ARTIFACT, output).run()
    assert result.status == "complete"
    patterns = np.load(ARTIFACT / "patterns.npz")
    with h5py.File(output / "run.h5", "r") as store:
        times = np.asarray(store["spikes/time_ms"])
        neurons = np.asarray(store["spikes/neuron"])
        cue = np.asarray(store["spikes/cue"])
        stored = np.asarray(store["rates/counts"])
        reconstructed = np.zeros_like(stored)
        bins = np.floor((times - 1e-7) / config.runtime.output_bin_ms).astype(int)
        modules = patterns["where"][neurons]
        np.add.at(reconstructed, (bins, modules), 1)
        np.testing.assert_array_equal(stored, reconstructed)
        assert cue.sum() >= 5
        assert store["statistics/window"].shape[0] == 4
        assert store["overlap/values"].shape == (4, config.network.patterns)
    analysis = analyze_run(output, make_figure=False)
    assert analysis["avalanche"]["count"] >= 1
    assert analysis["legacy_export"]["spikes"] == {
        "exported_legacy": 2,
        "legacy_limit": 2,
        "recorded_hdf5": len(times),
        "truncated": True,
    }
    legacy = output / "legacy"
    for name in ("spikes3-analysis-test.dat", "rate3-analysis-test.dat",
                 "temp3-analysis-test.dat", "medie3-analysis-test.dat",
                 "q3-analysis-test.dat"):
        assert (legacy / name).exists()
    assert (legacy / "spikes0-0-analysis-test.dat").exists()
    assert (legacy / "rate0-0-analysis-test.dat").exists()
    assert len((legacy / "spikes3-analysis-test.dat").read_text().splitlines()) == 2


def test_analysis_accepts_artifact_override_after_manifest_path_breaks(tmp_path):
    output = tmp_path / "run"
    config = analysis_config()
    SimulationRunner(config, ARTIFACT, output).run()
    copied_artifact = tmp_path / "new-location" / ARTIFACT.name
    import shutil

    shutil.copytree(ARTIFACT, copied_artifact)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_dir"] = str(tmp_path / "missing-artifact")
    manifest["artifact_relative"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = analyze_run(output, artifact_dir=copied_artifact, make_figure=False)
    assert Path(result["artifact_dir"]) == copied_artifact.resolve()


def test_rate_stop_is_evaluated_at_flush_boundary(tmp_path):
    config = analysis_config()
    config = replace(
        config,
        runtime=replace(config.runtime, duration_ms=10.0, chunk_ms=5.0),
        monitors=replace(config.monitors, flush_bins=1),
        execution=replace(
            config.execution,
            playback_ms=0.0,
            stop_rate_hz=0.001,
            stop_windows=1,
        ),
        cues=(CueConfig(pattern=0, start_ms=0.1, spike_count=1, frequency_hz=100.0),),
    )
    output = tmp_path / "stopped"
    result = SimulationRunner(config, ARTIFACT, output).run()
    assert result.status == "rate_stop"
    assert result.completed_step == 10
    with h5py.File(output / "run.h5", "r") as store:
        assert store["rates/counts"].shape[0] == 1
