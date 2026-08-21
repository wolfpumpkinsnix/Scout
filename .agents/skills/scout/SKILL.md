---
name: scout
description: "Use when working on Scout, the local document indexer: setup, GGUF embeddings, Docling chunking, LanceDB collections, and vector/FTS/hybrid search."
---

# Scout

Scout is a local document search engine. It converts documents with
Docling, chunks them with `HybridChunker`, embeds them with GGUF through
`llama-cpp-python`, and stores documents and vectors in LanceDB.

## Setup

Run from the project root on Windows:

```powershell
uv sync
```

Place an embedding model in `models/`, for example:

```text
models/Qwen3-Embedding-0.6B-Q8_0.gguf
```

For automatic GPU backend installation:

```powershell
.\scripts\install_llama_backend.ps1
```

The loader uses GPU offload automatically when the installed llama.cpp build
supports it and falls back to CPU otherwise.

## Ingest

Each ingest belongs to one logical collection:

```powershell
uv run python -m src.document_indexer ingest .\corpus-italia `
  --collection italia `
  --model .\models\Qwen3-Embedding-0.6B-Q8_0.gguf `
  --db-path data\lancedb
```

Collections share one LanceDB path. The collection name is stored on documents
and chunks, and is included in document identity to avoid collisions.

## Search

Hybrid search is the default:

```powershell
uv run python -m src.document_indexer query "regolamento CEE 880/92" `
  --db-path data\lancedb
```

Search one or more collections; omitting the option searches all collections:

```powershell
uv run python -m src.document_indexer query "bilancio" `
  --collection italia --collection azienda `
  --db-path data\lancedb
```

Available modes:

```powershell
uv run python -m src.document_indexer query "Ecolabel" --mode vector
uv run python -m src.document_indexer query "Ecolabel" --mode fts
uv run python -m src.document_indexer query "Ecolabel" --mode hybrid
```

`vector` uses semantic similarity, `fts` uses LanceDB full-text search, and
`hybrid` combines both ranked lists with Reciprocal Rank Fusion.

## Code map

- `src/document_indexer.py`: only application entry point and search/index logic.
- `src/indexer_support.py`: GGUF loading, embeddings, schemas, and LanceDB helpers.
- `src/logging_utils.py`: colored logging.
- `test/test_document_indexer.py`: persistence, collection, and search checks.

Keep the project GGUF-only. Do not add ONNX runtimes or model-specific
backends unless the project scope changes explicitly.

## Validation

```powershell
uv run python -m unittest -v test.test_document_indexer
uv run python -m src.document_indexer --help
```
