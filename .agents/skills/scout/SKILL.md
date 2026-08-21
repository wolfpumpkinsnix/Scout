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

## HTTP server

Keep the embedding model warm across client processes with the local server:

```powershell
uv run python -m src.document_indexer serve `
  --collection italia --db-path data\lancedb
```

Endpoints:

- `GET /health`: returns `{"status":"ok"}`.
- `GET /collections`: returns active collection names.
- `GET /documents` and `GET /documents/{id}`: list documents or retrieve
  document metadata and reconstructed text.
- `GET /status`: returns document, chunk, and collection counts.
- `POST /query`: accepts `query` (required), optional `collections` (string
  array), `mode` (`vector`, `fts`, or `hybrid`), `top_k`, and `rerank`.
- `POST /feedback`: accepts a `document_id`, `relevant` boolean, and optional
  query for ranking feedback.
- `POST /ingest`: accepts multipart `files` plus a `collection` field for
  drag-and-drop ingestion of one or more supported documents.
- `GET/POST/PUT/DELETE /collections[/{name}]`: manage the registered collection
  paths and file patterns in `index.yml`; `POST` also ingests matching files.
- `POST /update` (or JSON `POST /ingest`): ingest registered collections, optionally
  filtered with `{"collections":["italia"]}`.
- `DELETE /documents/{id}`: deactivate a document and its chunks.
- `POST /documents` with JSON `{path, collection}` indexes an existing local file;
  multipart remains available for drag-and-drop uploads.

Manage the registry from the CLI:

```powershell
uv run python -m src.document_indexer collection add .\corpus `
  --name italia --pattern "**/*.md"
uv run python -m src.document_indexer update --collection italia
uv run python -m src.document_indexer collection delete italia
```

The default `index.yml` uses JSON syntax, which is valid YAML, so the feature stays
stdlib-only.

`rerank` can be `true`, `false`, or `null`. Missing or `null` uses adaptive
reranking, the default. The server binds to `127.0.0.1:8181`; use `--host`
and `--port` to change it.

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
