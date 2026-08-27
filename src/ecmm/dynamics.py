from __future__ import annotations

from dataclasses import dataclass

import brainpy as bp
import brainpy.math as bm
import numpy as np
from scipy import sparse


class EventCSRLinear(bp.DynamicalSystem):
    """Event-driven J @ spikes using BrainPy's CSR operator.

    Offline artifacts store J as (post, pre). BrainPy connectors store rows as
    presynaptic neurons, so conversion to (pre, post) happens once at startup.
    """

    def __init__(self, post_by_pre: sparse.csr_matrix):
        super().__init__()
        if post_by_pre.shape[0] != post_by_pre.shape[1]:
            raise ValueError("The recurrent connectivity matrix must be square")
        pre_by_post = post_by_pre.transpose().tocsr()
        pre_by_post.sum_duplicates()
        pre_by_post.sort_indices()
        self.num = int(pre_by_post.shape[0])
        self.empty = pre_by_post.nnz == 0
        if self.empty:
            self.op = None
            return
        indices = np.asarray(pre_by_post.indices, dtype=np.int32)
        indptr = np.asarray(pre_by_post.indptr, dtype=np.int32)
        weights = np.asarray(pre_by_post.data, dtype=np.float32)
        connector = bp.conn.CSRConn(indices, indptr)
        connector(self.num, self.num)
        self.op = bp.dnn.EventCSRLinear(
            connector,
            weight=bm.asarray(weights),
            transpose=True,
        )

    def update(self, spikes):
        if self.empty:
            return bm.zeros(self.num, dtype=bm.float32)
        return self.op(spikes)


class DualExponentialLIF(bp.DynamicalSystem):
    """Fixed-step GPU form of the legacy A/B event-driven neuron state."""

    def __init__(
        self,
        num: int,
        dt_ms: float,
        tau_a_ms: float = 10.0,
        tau_b_ms: float = 5.0,
        threshold: float = 1.0,
    ):
        super().__init__()
        if min(num, dt_ms, tau_a_ms, tau_b_ms, threshold) <= 0:
            raise ValueError("Neuron size, time constants, dt and threshold must be positive")
        self.num = int(num)
        self.threshold = float(threshold)
        self.decay_a = float(np.exp(-dt_ms / tau_a_ms))
        self.decay_b = float(np.exp(-dt_ms / tau_b_ms))
        self.A = bm.Variable(bm.zeros(self.num, dtype=bm.float32))
        self.B = bm.Variable(bm.zeros(self.num, dtype=bm.float32))
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))

    def prepare(self, noise_kick):
        a = self.A.value * self.decay_a + noise_kick
        b = self.B.value * self.decay_b + noise_kick
        spike = (a - b) >= self.threshold
        return a, b, spike

    def commit(self, a, b, spike, recurrent_drive):
        self.A.value = bm.where(spike, 0.0, a) + recurrent_drive
        self.B.value = bm.where(spike, 0.0, b) + recurrent_drive
        self.spike.value = spike

    @property
    def voltage(self):
        return self.A.value - self.B.value

    def reset_state(self):
        self.A.value = bm.zeros_like(self.A.value)
        self.B.value = bm.zeros_like(self.B.value)
        self.spike.value = bm.zeros_like(self.spike.value)


class ECMMBrainPyNetwork(bp.DynamicalSystem):
    """BrainPy core retaining the legacy dual-exponential state equations."""

    def __init__(
        self,
        weights: sparse.csr_matrix,
        *,
        dt_ms: float,
        sigma: float,
        delta: float,
        tau_a_ms: float = 10.0,
        tau_b_ms: float = 5.0,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.num = int(weights.shape[0])
        self.sigma = float(sigma)
        self.delta = float(delta)
        self.neurons = DualExponentialLIF(
            self.num,
            dt_ms=dt_ms,
            tau_a_ms=tau_a_ms,
            tau_b_ms=tau_b_ms,
            threshold=threshold,
        )
        self.recurrent = EventCSRLinear(weights)

    def _step(self, noise_kick, cue_spike, sigma):
        a, b, spike = self.neurons.prepare(noise_kick)
        spike = bm.logical_or(spike, cue_spike)
        cue_flag = bm.logical_and(cue_spike, spike)
        excitation = sigma * self.recurrent(spike)
        spike_f = spike.astype(bm.float32)
        # The C++ loop inhibits every postsynaptic neuron except the firing one.
        inhibition = self.delta * (bm.sum(spike_f) - spike_f)
        self.neurons.commit(a, b, spike, excitation - inhibition)
        return spike, cue_flag

    def update(self, inputs):
        if isinstance(inputs, tuple):
            noise_kick, cue_spike, sigma = inputs
        else:
            noise_kick = inputs
            cue_spike = bm.zeros(self.num, dtype=bool)
            sigma = self.sigma
        spike, _cue_flag = self._step(noise_kick, cue_spike, sigma)
        return spike

    def step_with_metadata(self, noise_kick, cue_spike, sigma):
        return self._step(noise_kick, cue_spike, sigma)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "A": np.asarray(self.neurons.A.value, dtype=np.float32),
            "B": np.asarray(self.neurons.B.value, dtype=np.float32),
            "spike": np.asarray(self.neurons.spike.value, dtype=bool),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for key in ("A", "B", "spike"):
            if key not in state or np.shape(state[key]) != (self.num,):
                raise ValueError(f"Invalid or missing network state: {key}")
        self.neurons.A.value = bm.asarray(state["A"], dtype=bm.float32)
        self.neurons.B.value = bm.asarray(state["B"], dtype=bm.float32)
        self.neurons.spike.value = bm.asarray(state["spike"], dtype=bool)

    def reset_state(self):
        self.neurons.reset_state()


@dataclass(frozen=True)
class DynamicsSemantics:
    source: str = "legacy neuroni.c"
    integration: str = "synchronous fixed-step"
    voltage: str = "A-B"
    recurrent_orientation: str = "post_by_pre"
    simultaneous_spikes: str = "batched within dt"
