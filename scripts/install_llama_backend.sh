#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is for macOS." >&2
  exit 1
fi

export CMAKE_ARGS="${CMAKE_ARGS:-} -DGGML_METAL=on"
uv sync --locked --reinstall-package llama-cpp-python \
  --no-binary-package llama-cpp-python --no-cache

uv run python -c 'import sys, llama_cpp; from llama_cpp import llama_cpp as low; sys.exit("Metal verification failed: GPU offload is unavailable") if not llama_cpp.llama_supports_gpu_offload() else None; info = low.llama_print_system_info(); print(f"llama-cpp-python {llama_cpp.__version__}"); print(info.decode() if isinstance(info, bytes) else info); print("Metal backend verified.")'
