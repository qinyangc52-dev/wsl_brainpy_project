import brainpy.math as bm
import numpy as np
from scipy import sparse

from ecmm.models import ECMMBrainPyNetwork, LegacyEventReference, next_spike_delays


def test_analytic_next_spike_matches_legacy_quadratic():
    delay = next_spike_delays(np.array([5.0]), np.array([5.0]))[0]
    x = (5.0 + np.sqrt(25.0 - 20.0)) / 10.0
    np.testing.assert_allclose(delay, -10.0 * np.log(x))


def test_reference_emit_excludes_firing_neuron_from_global_inhibition():
    reference = LegacyEventReference(np.array([[0.0, 0.2], [0.3, 0.0]]), 2.0, 1.1)
    reference.emit(0)
    np.testing.assert_allclose(reference.A, [0.0, 2.0 * 0.3 - 1.1])
    np.testing.assert_allclose(reference.B, reference.A)


def test_fixed_step_spike_time_is_within_one_dt_of_event_reference():
    exact = next_spike_delays(np.array([5.0]), np.array([5.0]))[0]
    for dt in (0.2, 0.1, 0.05):
        model = ECMMBrainPyNetwork(
            sparse.csr_matrix((1, 1), dtype=np.float32),
            dt_ms=dt,
            sigma=0.0,
            delta=0.0,
        )
        model.neurons.A.value = bm.asarray([5.0])
        model.neurons.B.value = bm.asarray([5.0])
        observed = None
        for step in range(1000):
            spike = bool(np.asarray(model(bm.zeros(1)))[0])
            if spike:
                observed = (step + 1) * dt
                break
        assert observed is not None
        assert 0.0 <= observed - exact <= dt + 1e-6
