#!/usr/bin/env bash
set -euo pipefail

model="${1:-embedding}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models="$root/models"
mkdir -p "$models"

download() {
  local name="$1"
  local url="$2"
  local destination="$models/$name"
  if [[ -f "$destination" && "${FORCE:-0}" != "1" ]]; then
    echo "Already exists: $name"
    return
  fi
  echo "Downloading $name..."
  curl --fail --location --retry 3 --continue-at - --output "$destination" "$url"
}

case "$model" in
  embedding)
    download "Qwen3-Embedding-0.6B-Q8_0.gguf" \
      "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true"
    ;;
  gemma)
    download "embeddinggemma-300m-Q4_0.gguf" \
      "https://huggingface.co/second-state/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300m-Q4_0.gguf?download=true"
    ;;
  chat)
    download "Qwen3-1.7B-Q4_K_M.gguf" \
      "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf?download=true"
    ;;
  reranker)
    download "qwen3-reranker-0.6b-q8_0.gguf" \
      "https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf?download=true"
    ;;
  all)
    bash "$0" embedding
    bash "$0" gemma
    bash "$0" chat
    bash "$0" reranker
    ;;
  *)
    echo "Usage: $0 [embedding|gemma|chat|reranker|all]" >&2
    exit 2
    ;;
esac
