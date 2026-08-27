from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SIGMA_VALUES = tuple(Decimal(value) for value in ("6.85", "6.86", "6.87", "6.90", "6.95"))
SEEDS = tuple(range(1256874, 1256879))


@dataclass(frozen=True)
class ScanTask:
    sigma: Decimal
    seed: int
    run_name: str

    @property
    def sigma_tag(self) -> str:
        return f"{self.sigma:.2f}".replace(".", "p")


def build_tasks() -> list[ScanTask]:
    return [
        ScanTask(
            sigma=sigma,
            seed=seed,
            run_name=f"sigma_{str(sigma).replace('.', 'p')}_seed_{seed}",
        )
        for sigma in SIGMA_VALUES
        for seed in SEEDS
    ]


def is_complete(run_dir: Path) -> bool:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return summary.get("status") == "complete" and float(summary.get("completed_ms", 0.0)) >= 300000.0


def simulation_command(base_config: Path, task: ScanTask, artifact: Path, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ecmm.cli.main",
        "simulate",
        str(base_config),
        "--artifact",
        str(artifact),
        "--output",
        str(run_dir),
        "--set",
        f"runtime.sigma={task.sigma}",
        "--set",
        f"seeds.network={task.seed}",
        "--set",
        f"seeds.dynamics={task.seed}",
        "--set",
        f"artifact.name=full_seed_{task.seed}",
        "--set",
        f"io.run_name={task.run_name}",
    ]


def resume_command(task: ScanTask, artifact: Path, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ecmm.cli.main",
        "resume",
        str(run_dir),
        "--artifact",
        str(artifact),
    ]


def write_plan(path: Path, tasks: list[ScanTask], base_config: Path, run_root: Path) -> None:
    payload = {
        "design": "5 selected sigma points x 5 seeds",
        "sigma_start": float(SIGMA_VALUES[0]),
        "sigma_stop": float(SIGMA_VALUES[-1]),
        "sigma_values": [float(value) for value in SIGMA_VALUES],
        "seeds": list(SEEDS),
        "task_count": len(tasks),
        "duration_s_per_task": 300.0,
        "raw_sampling_hz": 1000.0,
        "base_config": str(base_config),
        "run_root": str(run_root),
        "tasks": [
            {"index": index, "sigma": float(task.sigma), "seed": task.seed, "run_name": task.run_name}
            for index, task in enumerate(tasks, start=1)
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_and_tee(command: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"COMMAND {shlex.join(command)}\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        return process.wait()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 25-task, 300 s selected sigma scan sequentially.")
    parser.add_argument("--base-config", type=Path, default=PROJECT / "configs" / "full_300s.yaml")
    parser.add_argument("--run-root", type=Path, default=PROJECT / "runs" / "fine_sigma_seed5_300s")
    parser.add_argument("--log-root", type=Path, default=PROJECT / "logs" / "fine_sigma_seed5_300s")
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_config = args.base_config.resolve()
    run_root = args.run_root.resolve()
    log_root = args.log_root.resolve()
    tasks = build_tasks()

    if not base_config.is_file():
        raise FileNotFoundError(f"Base configuration not found: {base_config}")
    missing_artifacts = [
        PROJECT / "artifacts" / f"full_seed_{seed}"
        for seed in SEEDS
        if not (PROJECT / "artifacts" / f"full_seed_{seed}" / "manifest.json").is_file()
    ]
    if missing_artifacts:
        raise FileNotFoundError(f"Missing formal artifacts: {missing_artifacts}")

    print(
        "PLAN sigma=[6.85,6.86,6.87,6.90,6.95], seeds=1256874..1256878, "
        f"tasks={len(tasks)}, duration=300 s, run_root={run_root}",
        flush=True,
    )
    if args.dry_run:
        for index, task in enumerate(tasks, start=1):
            artifact = PROJECT / "artifacts" / f"full_seed_{task.seed}"
            command = simulation_command(base_config, task, artifact, run_root / task.run_name)
            print(f"DRY-RUN {index:02d}/{len(tasks)} {shlex.join(command)}")
        return 0

    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    write_plan(run_root / "scan_plan.json", tasks, base_config, run_root)

    for index, task in enumerate(tasks, start=1):
        artifact = PROJECT / "artifacts" / f"full_seed_{task.seed}"
        run_dir = run_root / task.run_name
        log_path = log_root / f"{task.run_name}.log"

        if is_complete(run_dir):
            print(f"SKIP {index:02d}/{len(tasks)} complete: {task.run_name}", flush=True)
            continue
        if run_dir.exists():
            if not args.resume_incomplete:
                raise RuntimeError(
                    f"Incomplete run already exists: {run_dir}. "
                    "Inspect it, then rerun with --resume-incomplete to continue its checkpoint."
                )
            if not (run_dir / "checkpoint.npz").exists():
                raise RuntimeError(f"Incomplete run has no checkpoint and will not be overwritten: {run_dir}")
            command = resume_command(task, artifact, run_dir)
            action = "RESUME"
        else:
            command = simulation_command(base_config, task, artifact, run_dir)
            action = "START"

        print(
            f"{action} {index:02d}/{len(tasks)} sigma={task.sigma} seed={task.seed} output={run_dir}",
            flush=True,
        )
        return_code = run_and_tee(command, log_path)
        if return_code != 0:
            raise RuntimeError(f"Task failed with exit code {return_code}: {task.run_name}; log={log_path}")
        if not is_complete(run_dir):
            raise RuntimeError(f"Task exited without a complete 300 s summary: {run_dir}")
        print(f"DONE {index:02d}/{len(tasks)} {task.run_name}", flush=True)

    print(f"SCAN COMPLETE tasks={len(tasks)} run_root={run_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
