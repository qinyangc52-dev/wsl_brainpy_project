from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LegacyNetworkFixture:
    sites: np.ndarray
    H: np.ndarray
    who: np.ndarray
    phi: np.ndarray
    J: np.ndarray


def read_legacy_connection_file(
    path: str | Path, patterns: int, modules: int, neurons: int
) -> LegacyNetworkFixture:
    path = Path(path)
    with path.open("rb") as handle:
        sites = np.fromfile(handle, dtype="<i4", count=patterns * modules).reshape(patterns, modules)
        h = np.fromfile(handle, dtype="<i4", count=patterns)
        who = np.fromfile(handle, dtype="<i4", count=patterns * neurons).reshape(patterns, neurons)
        phi = np.fromfile(handle, dtype="<f8", count=patterns * neurons).reshape(patterns, neurons)
        weights = np.fromfile(handle, dtype="<f8", count=neurons * neurons).reshape(neurons, neurons)
        if handle.read(1):
            raise ValueError("Unexpected trailing data in legacy connection fixture")
    return LegacyNetworkFixture(sites, h, who, phi, weights)

