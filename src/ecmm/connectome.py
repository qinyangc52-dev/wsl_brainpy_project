from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np


_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _initializer_body(source: str, name: str) -> str:
    match = re.search(rf"double\s+{re.escape(name)}\s*\[[^;=]+\]\s*=\s*\{{", source)
    if match is None:
        raise ValueError(f"Cannot find C initializer for {name}")
    start = match.end() - 1
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    raise ValueError(f"Unterminated C initializer for {name}")


def extract_tractography_from_c(source_path: str | Path) -> dict[str, np.ndarray]:
    source = Path(source_path).read_text(encoding="utf-8")
    arrays: dict[str, np.ndarray] = {}
    expected = {"stmp": 66, "dtmp": 66 * 66, "wtmp": 66 * 66}
    for name, count in expected.items():
        values = np.asarray([float(x) for x in _NUMBER.findall(_initializer_body(source, name))])
        if values.size != count:
            raise ValueError(f"{name}: expected {count} values, found {values.size}")
        arrays[name] = values if name == "stmp" else values.reshape(66, 66)
    return arrays


def save_tractography(source_path: str | Path, output_path: str | Path) -> Path:
    arrays = extract_tractography_from_c(source_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        region_sizes=arrays["stmp"],
        distances=arrays["dtmp"],
        fiber_weights=arrays["wtmp"],
    )
    return output


def load_tractography(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["region_sizes"], data["distances"], data["fiber_weights"]


def effective_connectome(distances, fiber_weights, ddec: float, dmax: float) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    fibers = np.asarray(fiber_weights, dtype=np.float64)
    wmax = float(fibers.max())
    short_range = wmax * np.exp(-distances / ddec)
    return np.where(distances < dmax, fibers + short_range, fibers)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

