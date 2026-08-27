#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${ECMM_REMOTE_HOST:-sh01-ssh.gpuhome.cc}"
REMOTE_PORT="${ECMM_REMOTE_PORT:-30313}"
REMOTE_USER="${ECMM_REMOTE_USER:-root}"
REMOTE_ROOT="${ECMM_REMOTE_ROOT:-/root/rivermind-data/ecmm}"
SNAPSHOT="${ECMM_SNAPSHOT:-wsl_brainpy_project_remote_snapshot_20260821.tar.zst}"
EXPECTED_SHA256="${ECMM_SNAPSHOT_SHA256:-19f38ea9b4968f035ed1a4ccec834936e49f044a7e846c7633844dc21475398d}"
SSH_KEY="${ECMM_SSH_KEY:-$HOME/.ssh/ecmm_remote_ed25519}"
SYNC_ROOT="${ECMM_SYNC_ROOT:-$HOME/ecmm-sync}"
PARALLEL="${ECMM_PARALLEL:-8}"

PART_SIZE="32M"
REMOTE_PARTS="$REMOTE_ROOT/.ecmm_sync_parts_32m_20260821"
LOCAL_PARTS="$SYNC_ROOT/parts_32m"
LOCAL_ARCHIVE="$SYNC_ROOT/$SNAPSHOT"
EXTRACT_PROJECT="$SYNC_ROOT/remote_project"
LOCAL_PROJECT="${ECMM_LOCAL_PROJECT:-/mnt/c/SAO/Extended-Criticality--Modular-Model-main/wsl_brainpy_project}"
BACKUP_ROOT="${ECMM_BACKUP_ROOT:-/mnt/c/SAO/Extended-Criticality--Modular-Model-main/.codex_backups}"
SSH=(ssh -i "$SSH_KEY" -p "$REMOTE_PORT" -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

mkdir -p "$LOCAL_PARTS" "$SYNC_ROOT" "$BACKUP_ROOT"

"${SSH[@]}" "$REMOTE_USER@$REMOTE_HOST" \
  "set -e; mkdir -p '$REMOTE_PARTS'; if ! test -f '$REMOTE_PARTS/.complete'; then rm -f '$REMOTE_PARTS'/part-*; split -b '$PART_SIZE' -d -a 3 '$REMOTE_ROOT/$SNAPSHOT' '$REMOTE_PARTS/part-'; touch '$REMOTE_PARTS/.complete'; fi"

mapfile -t parts < <("${SSH[@]}" "$REMOTE_USER@$REMOTE_HOST" \
  "find '$REMOTE_PARTS' -maxdepth 1 -type f -name 'part-*' -printf '%f\n' | sort")

download_part() {
  local part="$1"
  local attempt
  for attempt in 1 2 3 4 5; do
    if rsync -a --partial --timeout=120 --info=progress2 \
      -e "ssh -i $SSH_KEY -p $REMOTE_PORT -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
      "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PARTS/$part" "$LOCAL_PARTS/$part"; then
      return 0
    fi
    sleep "$((attempt * 2))"
  done
  return 1
}
export -f download_part
export REMOTE_USER REMOTE_HOST REMOTE_PORT REMOTE_PARTS LOCAL_PARTS SSH_KEY
printf '%s\n' "${parts[@]}" | xargs -n 1 -P "$PARALLEL" bash -c 'download_part "$1"' _

cat "$LOCAL_PARTS"/part-* > "$LOCAL_ARCHIVE"
printf '%s  %s\n' "$EXPECTED_SHA256" "$LOCAL_ARCHIVE" | sha256sum --check --status

sync_base="$(realpath -m "$SYNC_ROOT")"
extract_project="$(realpath -m "$EXTRACT_PROJECT")"
case "$extract_project" in
  "$sync_base"/*) ;;
  *) printf 'Unsafe extraction target: %s\n' "$extract_project" >&2; exit 1 ;;
esac
rm -rf -- "$extract_project"
mkdir -p "$extract_project" "$LOCAL_PROJECT"
python3 "$(dirname "$0")/extract_zstd_tar.py" "$LOCAL_ARCHIVE" "$extract_project"

local_project="$(realpath -m "$LOCAL_PROJECT")"
case "$local_project" in
  /mnt/c/*/wsl_brainpy_project|/home/*/wsl_brainpy_project) ;;
  *) printf 'Unsafe local project target: %s\n' "$local_project" >&2; exit 1 ;;
esac
backup_dir="$(realpath -m "$BACKUP_ROOT/remote_sync_$(date +%Y%m%d_%H%M%S)")"
rsync -rt --backup --backup-dir="$backup_dir" "$extract_project/" "$local_project/"

printf 'Verified archive: %s\nExtracted project: %s\nUpdated project: %s\nBackup directory: %s\n' \
  "$LOCAL_ARCHIVE" "$extract_project" "$local_project" "$backup_dir"
