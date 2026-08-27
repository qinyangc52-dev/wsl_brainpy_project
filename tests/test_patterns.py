from pathlib import Path

import numpy as np

from ecmm.config import load_config
from ecmm.connectome import effective_connectome, load_tractography
from ecmm.legacy_rng import LegacyRNG
from ecmm.patterns import build_pattern_bank


PROJECT = Path(__file__).resolve().parents[1]


def test_prototype_pattern_bank_invariants():
    config = load_config(PROJECT / "configs" / "prototype.yaml")
    _, distances, fibers = load_tractography(PROJECT / "data" / "tractography_66.npz")
    weights = effective_connectome(distances, fibers, config.network.ddec, config.network.dmax)
    bank = build_pattern_bank(config.network, LegacyRNG(config.seeds.network), weights)

    assert bank.n_neurons == 1320
    np.testing.assert_array_equal(bank.H, [120, 120])
    assert bank.sites.shape == (2, 66)
    assert bank.who.shape == (2, 1320)
    for pattern in range(2):
        assert len(np.unique(bank.sites[pattern])) == 66
        assert len(np.unique(bank.who[pattern, : bank.H[pattern]])) == bank.H[pattern]
        assert np.all(np.diff(bank.phi[pattern, : bank.H[pattern]]) >= 0)
        assert np.all(bank.order[pattern, bank.who[pattern, : bank.H[pattern]]] >= 0)
        assert np.array_equal(np.sort(bank.posix[pattern]), np.arange(1320))

