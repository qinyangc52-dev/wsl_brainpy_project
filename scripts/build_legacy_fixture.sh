#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
PROJECT_DIR="$ROOT_DIR/wsl_brainpy_project"
RUN_DIR="$PROJECT_DIR/legacy_reference/prototype_cpp"

cd "$ROOT_DIR"
make
mkdir -p "$RUN_DIR/output"
cd "$RUN_DIR"
"$ROOT_DIR/a.out" > run.log 2>&1

test -f "$RUN_DIR/output/CONNESSIONI5-66-20-12-10-2-0-1-40-tract1"
grep -q "TIMING SUMMARY" run.log
echo "$RUN_DIR/output/CONNESSIONI5-66-20-12-10-2-0-1-40-tract1"

