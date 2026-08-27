import numpy as np

from ecmm.analysis import binned_rates, phase_overlap


def test_binned_module_rates():
    edges, rates = binned_rates(
        np.array([0.1, 0.9, 1.1]),
        np.array([0, 1, 2]),
        np.array([0, 0, 1]),
        np.array([2, 1]),
        duration_ms=2.0,
        bin_ms=1.0,
    )
    np.testing.assert_array_equal(edges, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(rates, [[1000.0, 0.0], [0.0, 1000.0]])


def test_phase_overlap_detects_matching_period():
    times = np.arange(0.0, 101.0, 20.0)
    neurons = np.zeros(len(times), dtype=np.int32)
    result = phase_overlap(
        times,
        neurons,
        order=np.array([[0]], dtype=np.int32),
        phi=np.array([[0.0]]),
        H=np.array([1]),
        period_min_ms=15.0,
        period_max_ms=30.0,
        period_step_ms=5.0,
    )
    assert result["best_period_ms"] == [20.0]
    np.testing.assert_allclose(result["overlap"], [1.0])
