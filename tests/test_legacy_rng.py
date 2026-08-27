from ecmm.legacy_rng import LegacyRNG


def test_legacy_rng_known_sequence():
    rng = LegacyRNG(1256878)
    values = [rng.uniform() for _ in range(4)]
    assert values == [
        0.06735913517703498,
        0.4500869286901661,
        0.8357210703568688,
        0.4266880834023902,
    ]


def test_legacy_rng_is_reproducible():
    left = LegacyRNG(12345)
    right = LegacyRNG(12345)
    assert [left.uniform() for _ in range(20)] == [right.uniform() for _ in range(20)]
    assert [left.normal() for _ in range(20)] == [right.normal() for _ in range(20)]
