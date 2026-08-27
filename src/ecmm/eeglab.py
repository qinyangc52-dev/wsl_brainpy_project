from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import resample_poly


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _channel_locations(labels: list[str]) -> np.ndarray:
    fields = [
        ("labels", object), ("type", object), ("theta", object),
        ("radius", object), ("X", object), ("Y", object), ("Z", object),
        ("sph_theta", object), ("sph_phi", object),
        ("sph_radius", object), ("urchan", object), ("ref", object),
    ]
    locations = np.empty((1, len(labels)), dtype=fields)
    for index, label in enumerate(labels):
        for field, _dtype in fields:
            locations[field][0, index] = np.empty((0, 0))
        locations["labels"][0, index] = label
        locations["type"][0, index] = "EEG"
        locations["urchan"][0, index] = float(index + 1)
    return locations


def _input_rate_hz(edges_ms: np.ndarray, rows: int) -> float:
    edges = np.asarray(edges_ms, dtype=np.float64).reshape(-1)
    if len(edges) != rows + 1:
        raise ValueError(
            f"rates/edges_ms has {len(edges)} values; expected {rows + 1}"
        )
    steps = np.diff(edges)
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError("rates/edges_ms must be finite and strictly increasing")
    step_ms = float(np.median(steps))
    if not np.allclose(steps, step_ms, rtol=1e-6, atol=1e-9):
        raise ValueError("rates/edges_ms is not uniformly sampled")
    return 1000.0 / step_ms


def export_run_to_eeglab(
    run_path: str | Path,
    output_dir: str | Path,
    *,
    filename_stem: str,
    target_sfreq_hz: float = 500.0,
    subject: str = "",
    condition: str = "simulation",
    session: int | None = None,
    channel_labels: list[str] | None = None,
    overwrite: bool = False,
) -> dict:
    """Export regional population firing rates from run.h5 to EEGLAB files.

    The source run remains unchanged. The exported channels are model regions,
    not scalp-voltage sensors. The external FDT stores little-endian float32
    samples in EEGLAB's channel-fastest order.
    """
    run_path = Path(run_path).resolve()
    if run_path.is_dir():
        run_path = run_path / "run.h5"
    if not run_path.is_file():
        raise FileNotFoundError(f"BrainPy run store not found: {run_path}")
    if target_sfreq_hz <= 0:
        raise ValueError("target_sfreq_hz must be positive")
    if not filename_stem or Path(filename_stem).name != filename_stem:
        raise ValueError("filename_stem must be one plain filename stem")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_path = output_dir / f"{filename_stem}.set"
    fdt_path = output_dir / f"{filename_stem}.fdt"
    manifest_path = output_dir / f"{filename_stem}.export.json"
    targets = (set_path, fdt_path, manifest_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Output already exists: " + ", ".join(existing))

    with h5py.File(run_path, "r") as store:
        if "rates/module_hz" not in store or "rates/edges_ms" not in store:
            raise KeyError("run.h5 must contain rates/module_hz and rates/edges_ms")
        source = np.asarray(store["rates/module_hz"], dtype=np.float32)
        edges = np.asarray(store["rates/edges_ms"], dtype=np.float64)
    if source.ndim != 2 or min(source.shape) == 0:
        raise ValueError("rates/module_hz must be a non-empty time-by-region matrix")
    if not np.all(np.isfinite(source)):
        raise ValueError("rates/module_hz contains NaN or infinite values")

    input_sfreq_hz = _input_rate_hz(edges, source.shape[0])
    ratio = Fraction(target_sfreq_hz / input_sfreq_hz).limit_denominator(10000)
    achieved_sfreq_hz = input_sfreq_hz * ratio.numerator / ratio.denominator
    if not np.isclose(achieved_sfreq_hz, target_sfreq_hz, rtol=1e-10):
        raise ValueError("The requested sampling-rate ratio is not representable")
    if ratio.numerator == ratio.denominator:
        exported = source
    else:
        exported = resample_poly(
            source,
            up=ratio.numerator,
            down=ratio.denominator,
            axis=0,
            padtype="line",
        ).astype(np.float32, copy=False)
    if not np.all(np.isfinite(exported)):
        raise ValueError("Resampled signal contains NaN or infinite values")

    n_points, n_channels = exported.shape
    labels = channel_labels or [f"ROI{index:02d}" for index in range(1, n_channels + 1)]
    if len(labels) != n_channels or len(set(labels)) != n_channels:
        raise ValueError("Channel labels must be unique and match the region count")
    if any(not str(label).strip() for label in labels):
        raise ValueError("Channel labels must be non-empty")

    temp_fdt = fdt_path.with_suffix(".fdt.tmp")
    temp_set = set_path.with_suffix(".set.tmp")
    for path in (temp_fdt, temp_set):
        if path.exists():
            path.unlink()
    # C-order time-by-channel bytes are channel-fastest, matching MATLAB's
    # column-major serialization of a channel-by-time EEG.data matrix.
    np.asarray(exported, dtype="<f4", order="C").tofile(temp_fdt)

    duration_seconds = n_points / target_sfreq_hz
    eeg = {
        "setname": filename_stem,
        "filename": set_path.name,
        "filepath": str(output_dir),
        "subject": str(subject),
        "group": "",
        "condition": str(condition),
        "session": np.empty((0, 0)) if session is None else float(session),
        "comments": (
            "BrainPy regional population firing rates exported from run.h5; "
            "channels are model regions, not scalp-voltage sensors."
        ),
        "nbchan": float(n_channels),
        "trials": 1.0,
        "pnts": float(n_points),
        "srate": float(target_sfreq_hz),
        "xmin": 0.0,
        "xmax": float((n_points - 1) / target_sfreq_hz),
        "times": np.arange(n_points, dtype=np.float64) * (1000.0 / target_sfreq_hz),
        "data": fdt_path.name,
        "icaact": np.empty((0, 0)),
        "icawinv": np.empty((0, 0)),
        "icasphere": np.empty((0, 0)),
        "icaweights": np.empty((0, 0)),
        "icachansind": np.empty((0, 0)),
        "chanlocs": _channel_locations([str(label) for label in labels]),
        "chaninfo": {"nosedir": "+X"},
        "ref": "unknown",
        "event": np.empty((0, 0)),
        "urevent": np.empty((0, 0)),
        "epoch": np.empty((0, 0)),
        "epochdescription": np.empty((0, 0)),
        "reject": {},
        "stats": {},
        "specdata": np.empty((0, 0)),
        "specicaact": np.empty((0, 0)),
        "saved": "yes",
    }
    savemat(temp_set, {"EEG": eeg}, appendmat=False, do_compression=True,
            long_field_names=True, oned_as="row")
    temp_fdt.replace(fdt_path)
    temp_set.replace(set_path)

    expected_bytes = n_points * n_channels * np.dtype("<f4").itemsize
    if fdt_path.stat().st_size != expected_bytes:
        raise RuntimeError("Written FDT size does not match exported dimensions")
    check = loadmat(set_path, squeeze_me=True, struct_as_record=False)["EEG"]
    if int(check.nbchan) != n_channels or int(check.pnts) != n_points:
        raise RuntimeError("Written SET metadata failed verification")
    if float(check.srate) != float(target_sfreq_hz) or str(check.data) != fdt_path.name:
        raise RuntimeError("Written SET sampling rate or FDT reference is invalid")

    manifest = {
        "schema_version": 1,
        "source_run_h5": str(run_path),
        "source_dataset": "/rates/module_hz",
        "signal_definition": (
            "Regional population firing rate in Hz; not scalp EEG voltage"
        ),
        "input_sampling_hz": input_sfreq_hz,
        "output_sampling_hz": float(target_sfreq_hz),
        "resampling": {
            "method": "scipy.signal.resample_poly",
            "up": ratio.numerator,
            "down": ratio.denominator,
            "anti_aliasing": True,
        },
        "channels": n_channels,
        "channel_labels": [str(label) for label in labels],
        "samples": n_points,
        "duration_seconds": duration_seconds,
        "set_file": set_path.name,
        "fdt_file": fdt_path.name,
        "fdt_dtype": "little-endian float32",
        "fdt_layout": "channel-fastest samples (EEGLAB external data)",
        "sha256": {"set": _sha256(set_path), "fdt": _sha256(fdt_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
