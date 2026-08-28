from dataclasses import replace
from pathlib import Path

import pytest

from ecmm.artifacts import artifact_rng_seeds, structural_hash
from ecmm.config import (
    ArtifactConfig,
    ConfigError,
    apply_overrides,
    config_from_dict,
    dump_config,
    load_config,
    parse_legacy_seed_text,
)
from ecmm.cli.main import build_parser


PROJECT = Path(__file__).resolve().parents[1]


def test_existing_prototype_uses_defaults_for_new_sections():
    config = load_config(PROJECT / "configs" / "prototype.yaml")
    assert config.network.total_neurons == 1320
    assert config.monitors.flush_bins == 10
    assert config.io.effective_output_dir == "output"
    assert config.seeds.effective_offline == 1256878


def test_full_configuration_contract_represents_original_scale():
    config = load_config(PROJECT / "configs" / "full.yaml")
    assert config.network.total_neurons == 13200
    assert config.network.patterns == 20
    assert config.runtime.duration_ms == 20000.0
    assert config.monitors.flush_bins == 30


def test_legacy_seed_and_cues_are_parsed():
    parsed = parse_legacy_seed_text(
        """seed=42 % comment
topo=random
S=4
Z=20
G=2
K=10
P=2
begin cue description
2
100
40
8
250
20
10
"""
    )
    assert parsed.values["seed"] == "42"
    assert len(parsed.cues) == 2
    assert parsed.cues[1].pattern == 1
    assert parsed.cues[1].start_ms == 250.0


def test_root_legacy_seed_maps_to_full_schema():
    config = load_config(PROJECT.parent / "SEED")
    assert config.network.total_neurons == 13200
    assert config.network.active_neurons_per_module == 100
    assert config.seeds.network == 1256874
    assert config.runtime.noise_mode == "gaussian"
    assert config.legacy_unmapped == {}


def test_dotted_override_is_typed_and_validated():
    config = load_config(PROJECT / "configs" / "prototype.yaml")
    updated = apply_overrides(config, {"runtime.sigma": 7.1, "io.run_name": "override"})
    assert updated.runtime.sigma == 7.1
    assert updated.io.run_name == "override"


def test_yaml_round_trip_preserves_contract(tmp_path):
    config = load_config(PROJECT / "configs" / "full.yaml")
    path = tmp_path / "roundtrip.yaml"
    dump_config(config, path)
    assert load_config(path) == config


def test_unknown_keys_and_invalid_dimensions_fail_early():
    with pytest.raises(ConfigError, match="Unknown network keys"):
        config_from_dict({"network": {"not_a_parameter": 1}})
    with pytest.raises(ConfigError, match="requires exactly 66"):
        config_from_dict({"network": {"topology": "tract1", "modules": 4}}).validate()


def test_per_module_size_vectors_are_supported():
    config = config_from_dict(
        {
            "network": {
                "topology": "random",
                "modules": 3,
                "neurons_per_module": [10, 20, 30],
                "active_modules_per_pattern": 2,
                "active_neurons_per_module": [5, 10, 15],
            }
        }
    ).validate()
    assert config.network.total_neurons == 60


def test_relocated_artifact_override_is_available_for_resume_and_analyze():
    parser = build_parser()
    resume = parser.parse_args(["resume", "runs/full", "--artifact", "artifacts/full"])
    analyze = parser.parse_args(["analyze", "runs/full", "--artifact", "artifacts/full"])
    assert resume.artifact == Path("artifacts/full")
    assert analyze.artifact == Path("artifacts/full")


def test_offline_artifact_seed_and_stream_affect_structural_identity():
    base = load_config(PROJECT / "configs" / "prototype.yaml")
    assert artifact_rng_seeds(base) == (base.seeds.network, 0)

    offline = replace(base, seeds=replace(base.seeds, offline=999, stream=17))
    assert artifact_rng_seeds(offline) == (999, 0)
    assert structural_hash(offline) != structural_hash(base)

    streamed = replace(base, seeds=replace(base.seeds, stream=17))
    assert artifact_rng_seeds(streamed) == (base.seeds.network, 17)
    assert structural_hash(streamed) != structural_hash(base)


@pytest.mark.parametrize("block_size", [0, -1, 1.5, True])
def test_invalid_stdp_block_size_fails_config_validation(block_size):
    base = load_config(PROJECT / "configs" / "prototype.yaml")
    invalid = replace(base, artifact=ArtifactConfig(
        name=base.artifact.name,
        dtype=base.artifact.dtype,
        stdp_block_size=block_size,
    ))
    with pytest.raises(ConfigError, match="stdp_block_size must be a positive integer"):
        invalid.validate()


@pytest.mark.parametrize(
    ("sigma_min", "sigma_max", "message"),
    [
        (5.0, None, "must be set together"),
        (None, 7.0, "must be set together"),
        (float("nan"), 7.0, "bounds must be finite"),
        (5.0, float("inf"), "bounds must be finite"),
        (8.0, 7.0, "cannot exceed"),
    ],
)
def test_invalid_sigma_schedule_fails_config_validation(sigma_min, sigma_max, message):
    base = load_config(PROJECT / "configs" / "prototype.yaml")
    invalid = replace(
        base,
        runtime=replace(base.runtime, sigma_min=sigma_min, sigma_max=sigma_max),
    )
    with pytest.raises(ConfigError, match=message):
        invalid.validate()


@pytest.mark.parametrize("field", ["output_bin_ms", "duration_ms", "chunk_ms"])
def test_runtime_grid_intervals_must_align_with_dt(field):
    base = load_config(PROJECT / "configs" / "prototype.yaml")
    invalid = replace(base, runtime=replace(base.runtime, **{field: 1.05}))
    with pytest.raises(ConfigError, match=rf"runtime\.{field} must be an integer multiple"):
        invalid.validate()
