from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import NetworkConfig
from .legacy_rng import LegacyRNG


@dataclass
class PatternBank:
    sites: np.ndarray
    who: np.ndarray
    phi: np.ndarray
    H: np.ndarray
    start: np.ndarray
    where: np.ndarray
    order: np.ndarray
    posix: np.ndarray
    Z: np.ndarray
    K: np.ndarray

    @property
    def n_neurons(self) -> int:
        return int(self.Z.sum())


def _module_sizes(config: NetworkConfig) -> tuple[np.ndarray, np.ndarray]:
    z_values, k_values = config.module_sizes()
    z = np.asarray(z_values, dtype=np.int32)
    k = np.asarray(k_values, dtype=np.int32)
    return z, k


def _build_sites(config: NetworkConfig, weights: np.ndarray | None, rng: LegacyRNG) -> np.ndarray:
    sites = np.empty((config.patterns, config.modules), dtype=np.int32)
    if config.topology == "random":
        for p in range(config.patterns):
            row = list(range(config.modules))
            if p > 0:
                rng.partial_permutation(row, config.modules)
            sites[p] = row
        return sites
    if config.topology != "tract1":
        raise NotImplementedError(f"Unsupported topology: {config.topology}")
    if weights is None or weights.shape != (config.modules, config.modules):
        raise ValueError("tract1 requires a square connectome matching the module count")

    totals = weights.sum(axis=1)
    for p in range(config.patterns):
        for attempt in range(10001):
            path = [rng.select_weighted(totals)]
            for i in range(1, config.active_modules_per_pattern + 1):
                previous = path[i - 1]
                while True:
                    candidate = rng.select_weighted(weights[previous])
                    if i == config.active_modules_per_pattern or candidate not in path:
                        break
                path.append(candidate)
            if path[config.active_modules_per_pattern] == path[0]:
                break
        else:
            raise RuntimeError(f"Cannot find a closed tractography path for pattern {p}")
        selected = path[: config.active_modules_per_pattern]
        remaining = [module for module in range(config.modules) if module not in selected]
        sites[p] = np.asarray(selected + remaining, dtype=np.int32)
    return sites


def build_pattern_bank(
    config: NetworkConfig,
    rng: LegacyRNG,
    connectome: np.ndarray | None = None,
) -> PatternBank:
    z, k = _module_sizes(config)
    n_neurons = int(z.sum())
    start = np.concatenate(([0], np.cumsum(z[:-1]))).astype(np.int32)
    where = np.repeat(np.arange(config.modules, dtype=np.int32), z)
    sites = _build_sites(config, connectome, rng)
    h = np.asarray(
        [sum(int(k[s]) for s in row[: config.active_modules_per_pattern]) for row in sites],
        dtype=np.int32,
    )

    who = np.full((config.patterns, n_neurons), -1, dtype=np.int32)
    phi = np.zeros((config.patterns, n_neurons), dtype=np.float64)
    for p in range(config.patterns):
        offset = 0
        for module in sites[p, : config.active_modules_per_pattern]:
            candidates = list(range(int(start[module]), int(start[module] + z[module])))
            if p > 0:
                rng.partial_permutation(candidates, int(k[module]))
            count = int(k[module])
            who[p, offset : offset + count] = candidates[:count]
            offset += count

    if config.sort == 0:
        for p in range(config.patterns):
            hp = int(h[p])
            for _ in range(int(round(config.swap * hp))):
                distance = int(math.floor(abs(config.range * rng.normal())))
                j1 = int(math.floor(hp * rng.uniform()))
                j2 = (j1 + distance) % hp
                who[p, j1], who[p, j2] = who[p, j2], who[p, j1]
            phase = np.asarray([2.0 * math.pi * rng.uniform() for _ in range(hp)])
            phi[p, :hp] = np.sort(phase)
    elif config.sort in (1, 2):
        for p in range(config.patterns):
            hp = int(h[p])
            offset = 0
            for module in sites[p, : config.active_modules_per_pattern]:
                count = int(k[module])
                if config.sort == 1:
                    p1 = 2.0 * math.pi * offset / hp
                    p2 = 2.0 * math.pi * (offset + count) / hp
                    values = [p1 + (p2 - p1) * rng.uniform() for _ in range(count)]
                else:
                    mean = 2.0 * math.pi * (offset + 0.5 * count) / hp
                    spread = 2.0 * math.pi * config.range * (0.5 * count) / hp
                    values = [mean + spread * rng.normal() for _ in range(count)]
                phi[p, offset : offset + count] = values
                offset += count
            permutation = np.argsort(phi[p, :hp], kind="heapsort")
            phi[p, :hp] = phi[p, :hp][permutation]
            who[p, :hp] = who[p, :hp][permutation]
    else:
        raise NotImplementedError(f"Unsupported sort mode: {config.sort}")

    order = np.full((config.patterns, n_neurons), -1, dtype=np.int32)
    posix = np.full((config.patterns, n_neurons), -1, dtype=np.int32)
    for p in range(config.patterns):
        hp = int(h[p])
        order[p, who[p, :hp]] = np.arange(hp, dtype=np.int32)
        position = 0
        active = np.zeros(n_neurons, dtype=bool)
        for module in range(config.modules):
            for neuron in who[p, :hp]:
                if where[neuron] == module:
                    active[neuron] = True
                    posix[p, neuron] = position
                    position += 1
            for neuron in range(int(start[module]), int(start[module] + z[module])):
                if not active[neuron]:
                    posix[p, neuron] = position
                    position += 1
        if position != n_neurons:
            raise RuntimeError("posix construction did not cover every neuron")

    return PatternBank(sites, who, phi, h, start, where, order, posix, z, k)
