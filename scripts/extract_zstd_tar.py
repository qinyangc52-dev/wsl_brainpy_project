#!/usr/bin/env python3
"""Safely extract a .tar.zst archive with the Python zstandard package."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import zstandard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    args.destination.mkdir(parents=True, exist_ok=True)
    with args.archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                archive.extractall(args.destination, filter="data")


if __name__ == "__main__":
    main()
