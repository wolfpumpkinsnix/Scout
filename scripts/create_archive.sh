#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d-%H%M%S)"
archive_dir="$root/Archive"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT

mkdir -p "$archive_dir"
rsync -a "$root/" "$staging/" \
  --exclude '.venv' --exclude 'build' --exclude 'dist' \
  --exclude 'models' --exclude 'data' --exclude 'Archive' \
  --exclude '__pycache__' --exclude '.git' --exclude '.copilot' \
  --exclude '*.pyc' --exclude '*.egg-info' \
  --exclude 'graphify-out'

# Keep Graphify knowledge in the shared archive when it exists.
if [[ -d "$root/graphify-out" ]]; then
  rsync -a "$root/graphify-out/" "$staging/graphify-out/"
fi

(cd "$staging" && zip -qr "$archive_dir/scout-$timestamp.zip" .)
echo "Archive created: $archive_dir/scout-$timestamp.zip"
