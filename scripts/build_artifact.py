import argparse
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ecmm.config import load_config
from ecmm.data import effective_connectome, load_tractography
from ecmm.offline import LegacyRNG, build_pattern_bank, build_stdp_csr, save_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "prototype.yaml")
    parser.add_argument("--tractography", type=Path, default=PROJECT / "data" / "tractography_66.npz")
    args = parser.parse_args()

    config = load_config(args.config)
    _, distances, fibers = load_tractography(args.tractography)
    connectome = effective_connectome(
        distances, fibers, config.network.ddec, config.network.dmax
    )
    rng = LegacyRNG(config.seeds.network)
    bank = build_pattern_bank(config.network, rng, connectome)
    weights = build_stdp_csr(
        bank,
        config.network.frequency_hz,
        block_size=config.artifact.stdp_block_size,
        dtype=config.artifact.dtype,
    )
    output = save_artifact(
        PROJECT / "artifacts" / config.artifact.name,
        config,
        args.tractography,
        bank,
        weights,
    )
    print(f"artifact={output}")
    print(f"shape={weights.shape} nnz={weights.nnz} dtype={weights.dtype}")


if __name__ == "__main__":
    main()
