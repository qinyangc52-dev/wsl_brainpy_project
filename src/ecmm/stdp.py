from __future__ import annotations

import math

import numpy as np
from scipy import sparse

from .patterns import PatternBank


def stdp_kernel(time_difference: np.ndarray) -> np.ndarray:
    t = np.asarray(time_difference, dtype=np.float64)
    return np.where(
        t > 0,
        0.4121037 * np.exp(-0.0980392 * t) - 0.2295345 * np.exp(-0.392157 * t),
        0.4121037 * np.exp(0.13986 * t) - 0.2295345 * np.exp(0.034965 * t),
    )


def build_stdp_csr(
    bank: PatternBank,
    frequency_hz: float,
    block_size: int = 256,
    dtype=np.float32,
) -> sparse.csr_matrix:
    if (
        not isinstance(block_size, (int, np.integer))
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")
    n_neurons = bank.n_neurons
    result = sparse.csr_matrix((n_neurons, n_neurons), dtype=dtype)
    period = 1000.0 / frequency_hz
    nmax = int(math.ceil(286.0 / period))

    for p in range(bank.who.shape[0]):
        hp = int(bank.H[p])
        neurons = bank.who[p, :hp]
        phases = bank.phi[p, :hp]
        pre_phase = phases[None, :]
        pre_ids = neurons[None, :]
        pattern_csr = sparse.csr_matrix((n_neurons, n_neurons), dtype=dtype)
        for start in range(0, hp, block_size):
            stop = min(start + block_size, hp)
            post_phase = phases[start:stop, None]
            phase_difference = (post_phase - pre_phase) / (2.0 * math.pi)
            values = np.zeros_like(phase_difference)
            for cycle in range(-nmax, nmax + 1):
                values += stdp_kernel(period * (cycle + phase_difference))

            local_post = np.arange(start, stop)[:, None]
            local_pre = np.arange(hp)[None, :]
            mask = local_post != local_pre
            rows = np.broadcast_to(neurons[start:stop, None], values.shape)[mask]
            cols = np.broadcast_to(pre_ids, values.shape)[mask]
            data = values[mask].astype(dtype, copy=False)
            block = sparse.coo_matrix(
                (data, (rows, cols)), shape=(n_neurons, n_neurons), dtype=dtype
            ).tocsr()
            pattern_csr += block
        result += pattern_csr

    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result
