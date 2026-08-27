#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from ecmm.config import load_config
from ecmm.runtime import SimulationRunner


PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first ECMM BrainPy/JAX prototype")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "prototype.yaml")
    parser.add_argument("--artifact", type=Path, default=PROJECT / "artifacts" / "prototype_seed_1256878")
    parser.add_argument("--output", type=Path, default=PROJECT / "runs" / "prototype_seed_1256878")
    parser.add_argument("--duration-ms", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.duration_ms is not None:
        config = replace(config, runtime=replace(config.runtime, duration_ms=args.duration_ms))
    result = SimulationRunner(config, args.artifact, args.output).run(resume=args.resume)
    print(result.output_dir / "summary.json")


if __name__ == "__main__":
    main()
