from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import CueConfig


_ASSIGNMENT = re.compile(r"^\s*([#A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


@dataclass(frozen=True)
class LegacySeed:
    values: dict[str, str]
    cues: tuple[CueConfig, ...]


def parse_legacy_seed(path: str | Path) -> LegacySeed:
    return parse_legacy_seed_text(Path(path).read_text(encoding="utf-8"))


def parse_legacy_seed_text(text: str) -> LegacySeed:
    lines = text.splitlines()
    values: dict[str, str] = {}
    cue_line: int | None = None
    for index, raw in enumerate(lines):
        clean = _strip_comment(raw).strip()
        if not clean:
            continue
        if clean.lower() == "begin cue description":
            cue_line = index + 1
            break
        match = _ASSIGNMENT.match(clean)
        if match:
            key = match.group(1).lstrip("#")
            values[key] = match.group(2).strip()
    cues = _parse_cues(lines[cue_line:]) if cue_line is not None else ()
    return LegacySeed(values=values, cues=cues)


def _strip_comment(line: str) -> str:
    return line.split("%", 1)[0]


def _parse_cues(lines: list[str]) -> tuple[CueConfig, ...]:
    tokens = [_strip_comment(line).strip() for line in lines]
    tokens = [token for token in tokens if token]
    if not tokens:
        return ()
    try:
        count = int(tokens[0])
    except ValueError as exc:
        raise ValueError("Cue description must start with an integer cue count") from exc
    expected = 1 + 3 * count
    if len(tokens) < expected:
        raise ValueError(f"Cue description requires {expected} values, found {len(tokens)}")
    cues = []
    for pattern in range(count):
        offset = 1 + 3 * pattern
        cues.append(
            CueConfig(
                pattern=pattern,
                start_ms=float(tokens[offset]),
                spike_count=int(tokens[offset + 1]),
                frequency_hz=float(tokens[offset + 2]),
            )
        )
    return tuple(cues)


def legacy_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
