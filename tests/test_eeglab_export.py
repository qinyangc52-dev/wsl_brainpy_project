from __future__ import annotations

import json

import h5py
import numpy as np
from scipy.io import loadmat

from ecmm.eeglab import export_run_to_eeglab


def test_export_run_to_eeglab_preserves_layout_and_metadata(tmp_path):
    run_path = tmp_path / "run.h5"
    source = np.arange(1000 * 3, dtype=np.float32).reshape(1000, 3)
    with h5py.File(run_path, "w") as store:
        store.create_dataset("rates/module_hz", data=source)
        store.create_dataset("rates/edges_ms", data=np.arange(1001, dtype=float))

    output = tmp_path / "eeglab"
    result = export_run_to_eeglab(
        run_path,
        output,
        filename_stem="sub-01_ses-1_simulated",
        target_sfreq_hz=500.0,
        subject="01",
        condition="A",
        session=1,
    )

    assert result["channels"] == 3
    assert result["samples"] == 500
    assert result["duration_seconds"] == 1.0
    fdt = np.fromfile(output / result["fdt_file"], dtype="<f4")
    assert fdt.size == 500 * 3
    time_by_channel = fdt.reshape(500, 3)
    assert np.all(np.isfinite(time_by_channel))
    assert not np.allclose(time_by_channel[:, 0], time_by_channel[:, 1])

    eeg = loadmat(output / result["set_file"], squeeze_me=True,
                  struct_as_record=False)["EEG"]
    assert int(eeg.nbchan) == 3
    assert int(eeg.pnts) == 500
    assert float(eeg.srate) == 500.0
    assert str(eeg.data) == result["fdt_file"]
    labels = [str(item.labels) for item in np.atleast_1d(eeg.chanlocs)]
    assert labels == ["ROI01", "ROI02", "ROI03"]

    manifest = json.loads(
        (output / "sub-01_ses-1_simulated.export.json").read_text()
    )
    assert manifest["signal_definition"].startswith("Regional population")
    assert manifest["resampling"]["anti_aliasing"] is True
