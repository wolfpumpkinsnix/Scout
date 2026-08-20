#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d-%H%M%S)"
archive_dir="$root/Archive"

mkdir -p "$archive_dir"
(cd "$root" && zip -qr "$archive_dir/scout-$timestamp.zip" . \
  -x '.venv/*' 'build/*' 'dist/*' 'models/*' 'data/*' 'Archive/*' \
  '__pycache__/*' '*/__pycache__/*' '.git/*' '.copilot/*' '*.pyc' '*.egg-info/*')
echo "Archive created: $archive_dir/scout-$timestamp.zip"
