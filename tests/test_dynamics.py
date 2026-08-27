import brainpy.math as bm
import numpy as np
from scipy import sparse

from ecmm.dynamics import ECMMBrainPyNetwork, EventCSRLinear


def test_event_csr_linear_preserves_post_by_pre_orientation():
    matrix = sparse.csr_matrix(
        np.array([[0.0, 2.0], [3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    )
    # Recurrent matrices are square; pad this hand-computable case to 3 x 3.
    matrix = sparse.hstack([matrix, sparse.csr_matrix((3, 1))]).tocsr()
    op = EventCSRLinear(matrix)
    actual = np.asarray(op(bm.asarray([True, False, True])))
    np.testing.assert_allclose(actual, matrix @ np.array([1.0, 0.0, 1.0]))


def test_dual_exponential_step_matches_equations_without_spike():
    matrix = sparse.csr_matrix((2, 2), dtype=np.float32)
    model = ECMMBrainPyNetwork(matrix, dt_ms=0.1, sigma=1.0, delta=0.0)
    spike = np.asarray(model(bm.asarray([0.5, 0.0], dtype=bm.float32)))
    np.testing.assert_array_equal(spike, [False, False])
    np.testing.assert_allclose(np.asarray(model.neurons.A), [0.5, 0.0])
    np.testing.assert_allclose(np.asarray(model.neurons.B), [0.5, 0.0])


def test_simultaneous_spikes_apply_global_inhibition_except_self():
    matrix = sparse.csr_matrix((3, 3), dtype=np.float32)
    model = ECMMBrainPyNetwork(
        matrix, dt_ms=0.1, sigma=0.0, delta=1.1, threshold=0.1
    )
    model.neurons.A.value = bm.asarray([1.0, 1.0, 0.0])
    spike = np.asarray(model(bm.zeros(3, dtype=bm.float32)))
    np.testing.assert_array_equal(spike, [True, True, False])
    np.testing.assert_allclose(np.asarray(model.neurons.A), [-1.1, -1.1, -2.2])
    np.testing.assert_allclose(np.asarray(model.neurons.B), [-1.1, -1.1, -2.2])
