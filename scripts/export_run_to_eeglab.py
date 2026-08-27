#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ecmm.eeglab import export_run_to_eeglab


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export BrainPy regional firing-rate time series from run.h5 to "
            "an EEGLAB .set/.fdt pair without modifying the source run."
        )
    )
    parser.add_argument("run", type=Path, help="run directory or run.h5 path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filename-stem", required=True)
    parser.add_argument("--target-sfreq", type=float, default=500.0)
    parser.add_argument("--subject", default="")
    parser.add_argument("--condition", default="simulation")
    parser.add_argument("--session", type=int)
    parser.add_argument(
        "--channel-labels",
        type=Path,
        help="optional UTF-8 text file containing one unique label per region",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    labels = None
    if args.channel_labels is not None:
        labels = [line.strip() for line in args.channel_labels.read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()]
    result = export_run_to_eeglab(
        args.run,
        args.output_dir,
        filename_stem=args.filename_stem,
        target_sfreq_hz=args.target_sfreq,
        subject=args.subject,
        condition=args.condition,
        session=args.session,
        channel_labels=labels,
        overwrite=args.overwrite,
    )
    print(f"SET={Path(args.output_dir).resolve() / result['set_file']}")
    print(f"FDT={Path(args.output_dir).resolve() / result['fdt_file']}")
    print(
        f"channels={result['channels']} samples={result['samples']} "
        f"sampling_hz={result['output_sampling_hz']} "
        f"duration_s={result['duration_seconds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
