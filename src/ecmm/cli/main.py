from __future__ import annotations

import argparse
from pathlib import Path

from ..config import dump_config, load_config, parse_override_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecmm", description="ECMM BrainPy project CLI")
    subcommands = parser.add_subparsers(dest="command", required=True)
    config = subcommands.add_parser("config", help="validate, inspect or convert configuration")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    for name in ("validate", "show"):
        command = config_sub.add_parser(name)
        command.add_argument("path", type=Path)
        command.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    convert = config_sub.add_parser("convert")
    convert.add_argument("path", type=Path, help="legacy SEED file")
    convert.add_argument("--output", "-o", type=Path, required=True)
    convert.add_argument("--strict", action="store_true")
    simulate = subcommands.add_parser("simulate", help="run a new BrainPy simulation")
    simulate.add_argument("config", type=Path)
    simulate.add_argument("--artifact", type=Path)
    simulate.add_argument("--output", type=Path)
    simulate.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    resume = subcommands.add_parser("resume", help="resume a checkpointed simulation")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--artifact", type=Path, help="override a relocated artifact directory")
    analyze = subcommands.add_parser("analyze", help="export legacy outputs and avalanche metrics")
    analyze.add_argument("run_dir", type=Path)
    analyze.add_argument("--artifact", type=Path, help="override a relocated artifact directory")
    analyze.add_argument("--no-figure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        from ..runtime import SimulationRunner

        config = load_config(args.config, parse_override_list(args.set))
        artifact = args.artifact or Path("artifacts") / config.artifact.name
        output = args.output or Path("runs") / config.io.run_name
        result = SimulationRunner(config, artifact, output).run()
        print(result.output_dir / "summary.json")
        return 0
    if args.command == "resume":
        import json
        from ..artifacts import resolve_run_artifact
        from ..runtime import SimulationRunner

        config = load_config(args.run_dir / "config.resolved.yaml")
        manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        artifact = resolve_run_artifact(args.run_dir, config, manifest, args.artifact)
        result = SimulationRunner(config, artifact, args.run_dir).run(resume=True)
        print(result.output_dir / "summary.json")
        return 0
    if args.command == "analyze":
        from ..analysis import analyze_run

        result = analyze_run(
            args.run_dir,
            artifact_dir=args.artifact,
            make_figure=not args.no_figure,
        )
        print(args.run_dir / "analysis" / "analysis_manifest.json")
        return 0
    if args.config_command == "convert":
        config = load_config(args.path, strict_legacy=args.strict)
        dump_config(config, args.output)
        print(args.output)
        return 0
    config = load_config(args.path, parse_override_list(args.set))
    if args.config_command == "show":
        print(dump_config(config), end="")
    else:
        print(
            f"valid: modules={config.network.modules} "
            f"neurons={config.network.total_neurons} patterns={config.network.patterns}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
