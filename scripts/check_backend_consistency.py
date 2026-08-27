#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import brainevent
import brainpy.math as bm
import jax
import numpy as np
from scipy import sparse

from ecmm.config import load_config
from ecmm.models import ECMMBrainPyNetwork


PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") == "cpu":
        brainevent.binary_csrmv_p.set_default("cpu", "jax_raw")
    config = load_config(PROJECT / "configs" / "prototype.yaml")
    runtime = config.runtime
    weights = sparse.load_npz(
        PROJECT / "artifacts" / config.artifact.name / "connectivity_csr.npz"
    ).astype(np.float32)
    model = ECMMBrainPyNetwork(
        weights,
        dt_ms=runtime.dt_ms,
        sigma=runtime.sigma,
        delta=runtime.delta,
        tau_a_ms=runtime.tau_a_ms,
        tau_b_ms=runtime.tau_b_ms,
        threshold=runtime.threshold,
    )
    rng = np.random.default_rng(20260819)
    # Backend comparison isolates dynamics from backend-specific random kernels.
    noise = rng.normal(0.0, 0.35, size=(100, weights.shape[0])).astype(np.float32)
    spikes = np.asarray(jax.device_get(bm.for_loop(model, bm.asarray(noise), jit=True)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        spikes=spikes,
        A=np.asarray(model.neurons.A.value),
        B=np.asarray(model.neurons.B.value),
        device=np.asarray(str(jax.devices()[0])),
    )
    print(jax.devices()[0], int(spikes.sum()), args.output)


if __name__ == "__main__":
    main()
