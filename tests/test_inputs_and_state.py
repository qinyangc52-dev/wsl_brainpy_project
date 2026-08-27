import brainpy.math as bm
import jax
import numpy as np
from scipy import sparse

from ecmm.config import CueConfig, RuntimeConfig
from ecmm.models import CueInput, ECMMBrainPyNetwork, NoiseInput, SigmaScheduler


def runtime(**updates):
    values = RuntimeConfig().__dict__ | updates
    return RuntimeConfig(**values)


def test_counter_noise_is_chunk_boundary_independent():
    source = NoiseInput.from_config(1234, runtime(alpha=0.5, rho=1.0, dt_ms=0.1))
    whole = np.asarray(source.chunk(0, 10, 8))
    pieces = np.concatenate(
        [np.asarray(source.chunk(0, 4, 8)), np.asarray(source.chunk(4, 6, 8))]
    )
    np.testing.assert_array_equal(whole, pieces)


def test_constant_noise_is_nonnegative_alpha_quantized():
    source = NoiseInput.from_config(
        9, runtime(alpha=0.25, rho=2.0, dt_ms=0.1, noise_mode="constant")
    )
    values = np.asarray(source.chunk(0, 20, 10))
    assert np.all(values >= 0)
    np.testing.assert_allclose(values / 0.25, np.round(values / 0.25))


def test_cue_schedule_reproduces_legacy_formula():
    who = np.array([[4, 7, 9, -1]], dtype=np.int32)
    source = CueInput.from_patterns(
        (CueConfig(pattern=0, start_ms=10.0, spike_count=4, frequency_hz=100.0),),
        who,
        np.array([3]),
        dt_ms=0.1,
        total_neurons=12,
    )
    assert source.neurons.tolist() == [4, 7, 9, 4]
    assert source.steps.tolist() == [99, 132, 166, 199]
    assert source.chunk(90, 20)[:, 4].sum() == 1


def test_sigma_scheduler_is_triangular():
    schedule = SigmaScheduler(
        sigma=6.0, duration_ms=10.0, dt_ms=1.0, sigma_min=5.0, sigma_max=7.0
    )
    values = np.asarray(schedule.values(0, 10))
    np.testing.assert_allclose(values[[0, 4, 9]], [5.4, 7.0, 5.0])


def test_cue_forces_spike_and_state_round_trip():
    model = ECMMBrainPyNetwork(
        sparse.csr_matrix((3, 3), dtype=np.float32),
        dt_ms=0.1,
        sigma=1.0,
        delta=0.0,
    )
    spike, cue = model.step_with_metadata(
        bm.zeros(3), bm.asarray([False, True, False]), bm.asarray(1.0)
    )
    np.testing.assert_array_equal(np.asarray(spike), [False, True, False])
    np.testing.assert_array_equal(np.asarray(cue), [False, True, False])
    saved = model.state_dict()
    model.reset_state()
    model.load_state_dict(saved)
    np.testing.assert_array_equal(model.state_dict()["A"], saved["A"])


def test_network_trajectory_is_chunk_boundary_independent():
    weights = sparse.csr_matrix(
        np.array([[0.0, 0.2], [0.3, 0.0]], dtype=np.float32)
    )
    model = ECMMBrainPyNetwork(weights, dt_ms=0.1, sigma=2.0, delta=0.1)
    noise = bm.asarray(np.random.default_rng(1).normal(0, 0.3, (20, 2)).astype(np.float32))
    cue = bm.zeros((20, 2), dtype=bool)
    sigma = bm.full((20,), 2.0, dtype=bm.float32)
    whole = bm.for_loop(model.step_with_metadata, (noise, cue, sigma), jit=True)
    whole = tuple(np.asarray(jax.device_get(value)) for value in whole)
    whole_state = model.state_dict()
    model.reset_state()
    first = bm.for_loop(
        model.step_with_metadata, (noise[:7], cue[:7], sigma[:7]), jit=True
    )
    second = bm.for_loop(
        model.step_with_metadata, (noise[7:], cue[7:], sigma[7:]), jit=True
    )
    joined = tuple(
        np.concatenate([np.asarray(jax.device_get(a)), np.asarray(jax.device_get(b))])
        for a, b in zip(first, second)
    )
    for expected, actual in zip(whole, joined):
        np.testing.assert_array_equal(actual, expected)
    for key, value in whole_state.items():
        np.testing.assert_array_equal(model.state_dict()[key], value)
