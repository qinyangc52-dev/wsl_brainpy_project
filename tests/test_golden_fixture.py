from pathlib import Path

import numpy as np
from scipy import sparse

from ecmm.legacy_fixture import read_legacy_connection_file


PROJECT = Path(__file__).resolve().parents[1]
FIXTURE = (
    PROJECT
    / "legacy_reference"
    / "prototype_cpp"
    / "output"
    / "CONNESSIONI5-66-20-12-10-2-0-1-40-tract1"
)
ARTIFACT = PROJECT / "artifacts" / "prototype_seed_1256878"


def test_python_artifact_matches_cpp_golden_fixture():
    assert FIXTURE.exists(), "Run scripts/build_legacy_fixture.sh first"
    assert ARTIFACT.exists(), "Run scripts/build_artifact.py first"

    golden = read_legacy_connection_file(FIXTURE, patterns=2, modules=66, neurons=1320)
    patterns = np.load(ARTIFACT / "patterns.npz")
    weights = sparse.load_npz(ARTIFACT / "connectivity_csr.npz")

    np.testing.assert_array_equal(patterns["sites"], golden.sites)
    np.testing.assert_array_equal(patterns["H"], golden.H)
    for pattern, count in enumerate(golden.H):
        np.testing.assert_array_equal(
            patterns["who"][pattern, :count], golden.who[pattern, :count]
        )
        np.testing.assert_array_equal(
            patterns["phi"][pattern, :count], golden.phi[pattern, :count]
        )

    assert weights.nnz == np.count_nonzero(golden.J)
    np.testing.assert_allclose(weights.toarray(), golden.J, rtol=1e-6, atol=1e-7)

