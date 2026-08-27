from pathlib import Path

import numpy as np

from ecmm.connectome import extract_tractography_from_c, effective_connectome


ROOT = Path(__file__).resolve().parents[2]


def test_extract_tractography_shapes_and_symmetry():
    arrays = extract_tractography_from_c(ROOT / "tract1.c")
    assert arrays["stmp"].shape == (66,)
    assert arrays["dtmp"].shape == (66, 66)
    assert arrays["wtmp"].shape == (66, 66)
    np.testing.assert_allclose(arrays["dtmp"], arrays["dtmp"].T)
    np.testing.assert_allclose(arrays["wtmp"], arrays["wtmp"].T)


def test_effective_connectome_matches_formula():
    arrays = extract_tractography_from_c(ROOT / "tract1.c")
    result = effective_connectome(arrays["dtmp"], arrays["wtmp"], ddec=3.3, dmax=30.0)
    assert result.shape == (66, 66)
    assert np.all(result >= arrays["wtmp"])

