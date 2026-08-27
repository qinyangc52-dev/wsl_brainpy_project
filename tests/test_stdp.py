import numpy as np

from ecmm.patterns import PatternBank
from ecmm.stdp import build_stdp_csr, stdp_kernel


def test_stdp_kernel_scalar_branches():
    values = stdp_kernel(np.array([-1.0, 1.0]))
    expected_negative = 0.4121037 * np.exp(-0.13986) - 0.2295345 * np.exp(-0.034965)
    expected_positive = 0.4121037 * np.exp(-0.0980392) - 0.2295345 * np.exp(-0.392157)
    np.testing.assert_allclose(values, [expected_negative, expected_positive])


def test_stdp_builder_keeps_zero_diagonal():
    bank = PatternBank(
        sites=np.array([[0]], dtype=np.int32),
        who=np.array([[0, 1]], dtype=np.int32),
        phi=np.array([[0.1, 1.2]], dtype=np.float64),
        H=np.array([2], dtype=np.int32),
        start=np.array([0], dtype=np.int32),
        where=np.array([0, 0], dtype=np.int32),
        order=np.array([[0, 1]], dtype=np.int32),
        posix=np.array([[0, 1]], dtype=np.int32),
        Z=np.array([2], dtype=np.int32),
        K=np.array([2], dtype=np.int32),
    )
    matrix = build_stdp_csr(bank, frequency_hz=8.0, block_size=1)
    assert matrix.nnz == 2
    np.testing.assert_array_equal(matrix.diagonal(), [0.0, 0.0])

