"""Document chunking, embedding, and LanceDB persistence."""
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportCallIssue=false, reportArgumentType=false

import argparse
import json
import logging
import math
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import lancedb
from lancedb.index import FTS
from llama_cpp import LLAMA_POOLING_TYPE_RANK
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from src.indexer_support import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    EMBEDDING_BATCH_SIZE,
    DOCLING_DOCUMENT_SUFFIXES,
    InputDocument,
    chunks_schema,
    documents_schema,
    embed_with_retries,
    hash_value,
    now,
    rows,
    upsert,
    load_model,
    read_input_documents,
)
from src.logging_utils import clear_progress, configure_colored_logging, update_progress

LOGGER = logging.getLogger("document-indexer")


@dataclass(frozen=True)
class DocumentIndexerConfig:
    model_path: Path = Path(DEFAULT_EMBEDDING_MODEL)
    db_path: Path = Path("data/lancedb")
    chunk_size: int = 900
    chunk_overlap: int | None = None
    batch_size: int = EMBEDDING_BATCH_SIZE
    gpu_layers: str | int = "auto"
    reranker_model_path: Path = Path(DEFAULT_RERANKER_MODEL)
    rerank_candidates: int = 12
    rerank_max_tokens: int = 1024
    reranker_context: int = 2048
    flash_attn: bool = False


@dataclass(frozen=True)
class DocumentChunk:
    document: InputDocument
    index: int
    text: str
    position: int


Row = dict[str, Any]

FTS_OPTIONS = {
    "base_tokenizer": "icu",
    "stem": False,
    "remove_stop_words": False,
    "lower_case": True,
    "ascii_folding": True,
}
LEXICAL_STOP_WORDS = {
    "a", "al", "alla", "and", "are", "be", "can", "che", "chi", "come",
    "cosa", "dei", "del", "della", "delle", "devono", "deve", "di", "do",
    "does", "dove", "e", "essere", "gli", "how", "i", "il", "in", "is",
    "la", "le", "lo", "must", "of", "on", "per", "perche", "perché",
    "possono", "puo", "può", "quale", "quali", "quando", "secondo", "should",
    "the", "to", "un", "una", "what", "when", "where", "which", "who", "why",
}
RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
RERANK_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
CHUNKER_VERSION = "articles-overlap-v2"
ARTICLE_HEADING = re.compile(r"(?m)^[ \t]*(Article[ \t]+\d+[A-Za-z]?)[ \t]*$")
OUTPUT_FIELDS = (
    "id", "document_id", "collection", "title", "relative_path", "chunk_index",
    "total_chunks", "position", "text", "_fts_rank", "_fts_score", "_vector_rank",
    "_vector_distance", "_hybrid_score", "_rerank_score",
)


def lexicalize_query(query: str) -> str:
    tokens = [token.text for token in lancedb.tokenize(query, **FTS_OPTIONS)]
    lexical = " ".join(token for token in tokens if token.casefold() not in LEXICAL_STOP_WORDS)
    return lexical or query


def _create_fts(table: Any) -> None:
    table.create_index("text", config=FTS(with_position=True, **FTS_OPTIONS), replace=True)


def load_reranker_model(path: Path, gpu_layers: str | int = "auto",
                        flash_attn: bool = False, context: int = 2048) -> Any:
    try:
        return load_model(str(path), embedding=True, gpu_layers=gpu_layers,
                          pooling_type=LLAMA_POOLING_TYPE_RANK, context=context,
                          flash_attn=flash_attn)
    except Exception as error:
        raise RuntimeError(
            f"Unable to load reranker {path}: {error}. Run "
            ".\\scripts\\download_models.ps1 -Model reranker"
        ) from error


def _reranker_prompt(query: str, document: str) -> str:
    return (
        f"<|im_start|>system\n{RERANK_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {RERANK_INSTRUCTION}\n\n"
        f"<Query>: {query}\n\n<Document>: {document}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _rerank(model: Any, query: str, candidates: list[Row],
            max_document_tokens: int = 1024) -> list[float]:
    started = time.perf_counter()
    context = int(model.n_ctx())
    model_batch = model.n_batch
    batch_capacity = min(context, int(model_batch() if callable(model_batch) else model_batch))
    documents = list(dict.fromkeys(str(candidate["text"]) for candidate in candidates))
    prompts: list[str] = []
    prompt_lengths: list[int] = []
    for document in documents:
        document_tokens = list(model.tokenize(
            document.encode("utf-8"), add_bos=False, special=False))
        if len(document_tokens) > max_document_tokens:
            document_tokens = document_tokens[:max_document_tokens]
            document = model.detokenize(document_tokens).decode("utf-8", errors="ignore")
        prompt = _reranker_prompt(query, document)
        tokens = model.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        if len(tokens) > batch_capacity:
            while len(tokens) > batch_capacity and document_tokens:
                document_tokens = document_tokens[:max(0, len(document_tokens) -
                                                        (len(tokens) - batch_capacity))]
                document = model.detokenize(document_tokens).decode("utf-8", errors="ignore")
                prompt = _reranker_prompt(query, document)
                tokens = model.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        if len(tokens) > batch_capacity:
            raise RuntimeError("query and reranker prompt exceed the model context window")
        prompts.append(prompt)
        prompt_lengths.append(len(tokens))
    outputs: list[Any] = []
    batch: list[str] = []
    batch_tokens = 0
    batch_count = 0
    inference_seconds = 0.0
    completed = 0
    update_progress("Reranking", 0, len(prompts))
    for prompt, token_count in zip(prompts, prompt_lengths):
        if batch and batch_tokens + token_count > batch_capacity:
            inference_started = time.perf_counter()
            outputs.extend(model.embed(batch, normalize=False, truncate=False))
            inference_seconds += time.perf_counter() - inference_started
            batch_count += 1
            completed += len(batch)
            update_progress("Reranking", completed, len(prompts))
            batch, batch_tokens = [], 0
        batch.append(prompt)
        batch_tokens += token_count
    if batch:
        inference_started = time.perf_counter()
        outputs.extend(model.embed(batch, normalize=False, truncate=False))
        inference_seconds += time.perf_counter() - inference_started
        batch_count += 1
        update_progress("Reranking", len(prompts), len(prompts))
    if len(outputs) != len(documents):
        raise RuntimeError("incompatible reranker output: wrong batch size")
    scores: list[float] = []
    for output in outputs:
        if not isinstance(output, Sequence) or len(output) < 2:
            raise RuntimeError("incompatible reranker output: expected two rank values")
        yes, no = float(output[0]), float(output[1])
        if (not all(map(math.isfinite, (yes, no))) or not 0 <= yes <= 1 or
                not 0 <= no <= 1 or abs(yes + no - 1) > 0.05):
            raise RuntimeError(
                "incompatible reranker output: rank values must be finite, in [0, 1], "
                "and approximately complementary"
            )
        scores.append(yes)
    scores_by_document = dict(zip(documents, scores))
    total_tokens = sum(prompt_lengths)
    LOGGER.info(
        "phase=rerank-runtime candidates=%d unique=%d batches=%d total_tokens=%d "
        "max_prompt_tokens=%d inference_seconds=%.3f tokens_per_second=%.1f seconds=%.3f",
        len(candidates), len(documents), batch_count, total_tokens,
        max(prompt_lengths, default=0), inference_seconds,
        total_tokens / inference_seconds if inference_seconds else 0.0,
        time.perf_counter() - started,
    )
    return [scores_by_document[str(candidate["text"])] for candidate in candidates]


def load_embedding_model(path: Path, gpu_layers: str | int = "auto",
                         flash_attn: bool = False) -> Any:
    return load_model(str(path), embedding=True, gpu_layers=gpu_layers,
                      flash_attn=flash_attn)


def _pdf_sections(text: str) -> list[tuple[int, int, str]]:
    matches = list(ARTICLE_HEADING.finditer(text))
    if not matches:
        return [(0, len(text), text)]
    sections: list[tuple[int, int, str]] = []
    if text[:matches[0].start()].strip():
        sections.append((0, matches[0].start(), text[:matches[0].start()]))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        markdown = f"# {match.group(1)}\n\n{body}" if body else f"# {match.group(1)}"
        sections.append((match.start(), end, markdown))
    return sections


def _whitespace_position(text: str, needle: str, start: int, end: int) -> int:
    collapsed: list[str] = []
    offsets: list[int] = []
    whitespace: int | None = None
    for offset in range(start, end):
        character = text[offset]
        if character.isspace():
            if collapsed and whitespace is None:
                whitespace = offset
            continue
        if whitespace is not None:
            collapsed.append(" ")
            offsets.append(whitespace)
            whitespace = None
        collapsed.append(character)
        offsets.append(offset)
    match = "".join(collapsed).find(" ".join(needle.split()))
    return offsets[match] if match >= 0 and offsets else -1


def _chunk_position(text: str, needle: str, start: int, end: int,
                    fallback: int, path: str) -> int:
    position = text.find(needle, start, end)
    if position < 0:
        position = _whitespace_position(text, needle, start, end)
    if position < 0:
        anchor = " ".join(needle.split()[:12])
        position = _whitespace_position(text, anchor, start, end)
        if position >= 0:
            LOGGER.info("phase=chunk-position-anchor document=%s position=%d", path, position)
    if position < 0:
        LOGGER.warning("phase=chunk-position-fallback document=%s section=%d", path, fallback)
        return fallback
    return position


def _overlap_suffix(text: str, tokenizer: Any, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    for pattern in (r"\n+|(?<=[.!?])\s+", r"\s+"):
        for match in re.finditer(pattern, text):
            candidate = text[match.end():]
            if candidate and tokenizer.count_tokens(candidate) <= max_tokens:
                return candidate
    return text if tokenizer.count_tokens(text) <= max_tokens else ""


def chunk_documents(
    documents: list[InputDocument], model: Any, chunk_size: int, overlap_tokens: int
) -> list[DocumentChunk]:
    del model
    converter = DocumentConverter()
    chunker = HybridChunker()
    docling_tokenizer: Any = chunker.tokenizer
    docling_tokenizer.max_tokens = chunk_size - overlap_tokens
    chunks: list[DocumentChunk] = []
    update_progress("Chunking", 0, len(documents))
    for document_number, document in enumerate(documents, start=1):
        source = Path(document.collection_root) / document.relative_path
        sections = (_pdf_sections(document.text) if source.suffix.lower() == ".pdf"
                    else [(0, len(document.text), "")])
        document_chunks: list[Any] = []
        for section_start, section_end, markdown in sections:
            converted = (converter.convert_string(
                markdown or document.text, InputFormat.MD, name=source.stem)
                if source.suffix.lower() == ".pdf"
                or source.suffix.lower() in DOCLING_DOCUMENT_SUFFIXES
                else converter.convert(source))
            section_chunks = list(chunker.chunk(converted.document))
            previous_text = ""
            previous_position = section_start
            for chunk in section_chunks:
                search_start = section_start if not previous_text else previous_position + 1
                chunk_position = _chunk_position(
                    document.text, chunk.text, search_start, section_end,
                    search_start, document.relative_path)
                overlap = _overlap_suffix(previous_text, docling_tokenizer, overlap_tokens)
                text = chunker.contextualize(chunk)
                stored_position = chunk_position
                if overlap:
                    stored_position = _chunk_position(
                        document.text, overlap, previous_position, chunk_position,
                        chunk_position, document.relative_path)
                    text = f"{text[:-len(chunk.text)]}{overlap}\n{chunk.text}"
                chunks.append(DocumentChunk(
                    document, len(document_chunks), text, stored_position))
                document_chunks.append(chunk)
                previous_text = chunk.text
                previous_position = chunk_position
        LOGGER.info("phase=docling-chunk document=%s chunks=%d",
                    document.relative_path, len(document_chunks))
        update_progress("Chunking", document_number, len(documents))
    return chunks


class DocumentIndexer:
    def __init__(self, config: DocumentIndexerConfig = DocumentIndexerConfig()) -> None:
        if config.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if config.chunk_overlap is not None and not 0 <= config.chunk_overlap < config.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if config.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if config.rerank_candidates < 1:
            raise ValueError("rerank_candidates must be positive")
        if config.rerank_max_tokens < 1:
            raise ValueError("rerank_max_tokens must be positive")
        if config.reranker_context < 1:
            raise ValueError("reranker_context must be positive")
        self.config = config
        self._reranker: Any | None = None

    def index(
        self, path: Path, model: Any | None = None, collection: str = "default"
    ) -> dict[str, int | str]:
        started = time.perf_counter()
        if not collection.strip():
            raise ValueError("collection must not be empty")
        if not path.exists():
            raise FileNotFoundError(f"Document path not found: {path}")
        LOGGER.info("phase=scan path=%s", path)
        documents = read_input_documents(path)
        if not documents:
            raise ValueError(f"No supported documents found: {path}")
        LOGGER.info("phase=scan-complete documents=%d", len(documents))
        chunk_size = min(self.config.chunk_size, 1800) \
            if "gemma" in str(self.config.model_path).lower() else self.config.chunk_size
        chunk_overlap = (self.config.chunk_overlap if self.config.chunk_overlap is not None
                         else round(chunk_size * 0.15))
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than the effective chunk_size")
        db: Any = lancedb.connect(str(self.config.db_path))
        table_names = db.list_tables().tables
        documents_table: Any = (
            db.open_table("documents")
            if "documents" in table_names
            else db.create_table("documents", schema=documents_schema())
        )
        embedding_fingerprint = (
            f"{self.config.model_path.resolve()}|chunker={CHUNKER_VERSION}"
            f"|chunk_size={chunk_size}|chunk_overlap={chunk_overlap}"
        )
        document_ids = {
            document.id: hash_value([collection, document.id]) for document in documents
        }
        existing_documents = {
            row["id"]: row for row in documents_table.to_arrow().to_pylist()
        }
        chunks_table: Any | None = db.open_table("chunks") if "chunks" in table_names else None
        pending_documents: list[InputDocument] = []
        for document in documents:
            existing = existing_documents.get(document_ids[document.id])
            active_chunks = 0 if chunks_table is None else chunks_table.count_rows(
                f"document_id = {document_ids[document.id]!r} AND active = true"
            )
            unchanged = bool(
                existing and existing.get("active")
                and existing.get("content_hash") == document.content_hash
                and existing.get("embedding_fingerprint") == embedding_fingerprint
                and int(existing.get("total_chunks") or 0) > 0
                and active_chunks == int(existing["total_chunks"])
            )
            if not unchanged:
                pending_documents.append(document)

        deleted_rows: list[Row] = []
        if path.is_dir():
            root = documents[0].collection_root
            current_ids = set(document_ids.values())
            deleted_rows = [
                row for row in existing_documents.values()
                if row.get("active") and row.get("collection") == collection
                and row.get("collection_root") == root and row["id"] not in current_ids
            ]
            upsert(documents_table, [{**row, "active": False} for row in deleted_rows])
            if chunks_table is not None:
                for row in deleted_rows:
                    old_chunks = rows(chunks_table, f"document_id = {row['id']!r}")
                    upsert(chunks_table, [{**chunk, "active": False} for chunk in old_chunks])

        LOGGER.info("phase=incremental-scan changed=%d unchanged=%d deleted=%d",
                    len(pending_documents), len(documents) - len(pending_documents),
                    len(deleted_rows))
        if not pending_documents:
            if chunks_table is not None and deleted_rows:
                fts_started = time.perf_counter()
                _create_fts(chunks_table)
                LOGGER.info("phase=fts-index-complete seconds=%.3f",
                            time.perf_counter() - fts_started)
            result: dict[str, int | str] = {
                "documents": len(documents), "chunks": 0, "embedded": 0,
                "failures": 0, "unchanged": len(documents),
                "deleted": len(deleted_rows), "db_path": str(self.config.db_path),
            }
            LOGGER.info("phase=complete documents=%d chunks=0 embedded=0 failures=0 "
                        "unchanged=%d deleted=%d seconds=%.3f", len(documents),
                        len(documents), len(deleted_rows), time.perf_counter() - started)
            return result

        model_started = time.perf_counter()
        embedding_model: Any = model if model is not None else load_embedding_model(
            self.config.model_path, self.config.gpu_layers, self.config.flash_attn)
        LOGGER.info("phase=load-model-complete seconds=%.3f", time.perf_counter() - model_started)
        dimension = int(embedding_model.n_embd())
        if chunks_table is None:
            chunks_table = db.create_table("chunks", schema=chunks_schema(dimension))
        else:
            current_dimension = chunks_table.schema.field("vector").type.list_size
            if current_dimension != dimension:
                raise ValueError(
                    f"Existing LanceDB vector dimension is {current_dimension}, expected {dimension}"
                )
        LOGGER.info("phase=database-ready path=%s dimension=%d", self.config.db_path, dimension)
        chunk_started = time.perf_counter()
        chunks = chunk_documents(
            pending_documents, embedding_model, chunk_size, chunk_overlap)
        LOGGER.info("phase=chunk-complete chunks=%d seconds=%.3f",
                    len(chunks), time.perf_counter() - chunk_started)

        rows_by_document: dict[str, list[Row]] = {
            document.id: [] for document in pending_documents
        }
        total_chunks_by_document = {document.id: 0 for document in documents}
        for chunk in chunks:
            total_chunks_by_document[chunk.document.id] += 1
        document_rows: list[Row] = []
        old_chunks_by_document: dict[str, list[Row]] = {}
        generation_by_document = {
            document.id: hash_value([document.content_hash, embedding_fingerprint, now()])
            for document in pending_documents
        }
        for document in pending_documents:
            document_started = time.perf_counter()
            expected_chunks = total_chunks_by_document[document.id]
            LOGGER.info("phase=prepare document=%d/%d path=%s chunks=%d",
                        len(document_rows) + 1, len(pending_documents),
                        document.relative_path, expected_chunks)
            document_id = document_ids[document.id]
            old_chunks_by_document[document.id] = rows(
                chunks_table, f"document_id = {document_id!r}"
            )
            document_rows.append({
                "id": document_id,
                "collection_root": document.collection_root,
                "collection": collection,
                "relative_path": document.relative_path,
                "content_hash": document.content_hash,
                "embedding_fingerprint": embedding_fingerprint,
                "title": Path(document.relative_path).stem,
                "total_chunks": total_chunks_by_document[document.id],
                "active": False,
                "updated_at": now(),
            })
            LOGGER.info("phase=prepare-complete document=%d/%d seconds=%.3f",
                        len(document_rows), len(documents),
                        time.perf_counter() - document_started)
        failures = 0
        batch_count = (len(chunks) + self.config.batch_size - 1) // self.config.batch_size
        update_progress("Embedding", 0, len(chunks))
        for start in range(0, len(chunks), self.config.batch_size):
            batch = chunks[start:start + self.config.batch_size]
            batch_started = time.perf_counter()
            texts = [chunk.text for chunk in batch]
            batch_number = start // self.config.batch_size + 1
            batch_documents = sorted({chunk.document.relative_path for chunk in batch})
            LOGGER.info("phase=embed-start batch=%d/%d chunks=%d documents=%d chars=%d",
                        batch_number, batch_count, len(batch), len(batch_documents),
                        sum(len(text) for text in texts))
            vectors, batch_failures = embed_with_retries(embedding_model, texts)
            failures += batch_failures
            batch_rows: list[Row] = []
            for chunk, vector in zip(batch, vectors):
                if vector is None:
                    continue
                row: Row = {
                    "id": hash_value([
                        chunk.document.id, generation_by_document[chunk.document.id], chunk.index
                    ]),
                    "document_id": document_ids[chunk.document.id],
                    "collection": collection,
                    "content_hash": chunk.document.content_hash,
                    "embedding_fingerprint": embedding_fingerprint,
                    "chunk_index": chunk.index,
                    "total_chunks": total_chunks_by_document[chunk.document.id],
                    "position": chunk.position,
                    "text": chunk.text,
                    "vector": vector,
                    "active": False,
                }
                rows_by_document[chunk.document.id].append(row)
                batch_rows.append(row)
            upsert_started = time.perf_counter()
            upsert(chunks_table, batch_rows)
            elapsed = time.perf_counter() - batch_started
            LOGGER.info("phase=embed-complete batch=%d/%d chunks=%d embedded=%d failures=%d "
                        "embed_seconds=%.3f upsert_seconds=%.3f total_seconds=%.3f",
                        batch_number, batch_count, len(batch), len(batch_rows),
                        batch_failures, elapsed - (time.perf_counter() - upsert_started),
                        time.perf_counter() - upsert_started, elapsed)
            update_progress("Embedding", min(start + len(batch), len(chunks)), len(chunks))

        active_documents: list[Row] = []
        active_chunks: list[Row] = []
        obsolete_chunks: list[Row] = []
        document_rows_by_id: dict[str, Row] = {
            document.id: row for document, row in zip(pending_documents, document_rows)
        }
        for document in pending_documents:
            document_chunk_rows = rows_by_document[document.id]
            expected = total_chunks_by_document[document.id]
            if len(document_chunk_rows) == expected and expected:
                active_documents.append({**document_rows_by_id[document.id], "active": True})
                active_chunks.extend([{**row, "active": True} for row in document_chunk_rows])
                obsolete_chunks.extend([
                    {**row, "active": False}
                    for row in old_chunks_by_document[document.id]
                ])
        upsert(documents_table, active_documents)
        upsert(chunks_table, active_chunks)
        upsert(chunks_table, obsolete_chunks)
        fts_started = time.perf_counter()
        _create_fts(chunks_table)
        LOGGER.info("phase=fts-index-complete seconds=%.3f", time.perf_counter() - fts_started)
        embedded = sum(len(document_rows) for document_rows in rows_by_document.values())
        result: dict[str, int | str] = {
            "documents": len(documents),
            "chunks": len(chunks),
            "embedded": embedded,
            "failures": failures,
            "unchanged": len(documents) - len(pending_documents),
            "deleted": len(deleted_rows),
            "db_path": str(self.config.db_path),
        }
        LOGGER.info("phase=complete documents=%d chunks=%d embedded=%d failures=%d "
                    "unchanged=%d deleted=%d seconds=%.3f",
                    len(documents), len(chunks), embedded, failures,
                    len(documents) - len(pending_documents), len(deleted_rows),
                    time.perf_counter() - started)
        return result

    def reindex_fts(self) -> dict[str, int | str]:
        started = time.perf_counter()
        db: Any = lancedb.connect(str(self.config.db_path))
        if "chunks" not in db.list_tables().tables:
            raise ValueError("database is empty; ingest documents before rebuilding FTS")
        table: Any = db.open_table("chunks")
        _create_fts(table)
        result: dict[str, int | str] = {
            "db_path": str(self.config.db_path),
            "indexed_chunks": table.count_rows(),
        }
        LOGGER.info("phase=fts-reindex-complete chunks=%d seconds=%.3f",
                    result["indexed_chunks"], time.perf_counter() - started)
        return result

    def list_collections(self) -> list[str]:
        db: Any = lancedb.connect(str(self.config.db_path))
        if "documents" not in db.list_tables().tables:
            return []
        return sorted({
            str(row["collection"])
            for row in db.open_table("documents").to_arrow().to_pylist()
            if row.get("active") and row.get("collection")
        })

    def search(
        self,
        query: str,
        collections: list[str] | None = None,
        mode: str = "vector",
        top_k: int = 5,
        model: Any | None = None,
        rerank: bool | None = None,
        reranker: Any | None = None,
    ) -> list[Row]:
        started = time.perf_counter()
        if not query.strip():
            raise ValueError("query must not be empty")
        if mode not in {"vector", "fts", "hybrid"}:
            raise ValueError("mode must be 'vector', 'fts' or 'hybrid'")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        LOGGER.info("phase=query-start mode=%s collections=%s top_k=%d",
                    mode, collections or "all", top_k)
        db: Any = lancedb.connect(str(self.config.db_path))
        if "chunks" not in db.list_tables().tables:
            raise ValueError("database is empty; ingest documents before querying")
        table: Any = db.open_table("chunks")
        filters = ["active = true"]
        if collections:
            if "collection" not in table.schema.names:
                raise ValueError("database was created before collection support; re-index it")
            values = ",".join(repr(value) for value in collections)
            filters.append(f"collection IN ({values})")
        where = " AND ".join(filters)

        def enrich(results: list[Row]) -> list[Row]:
            documents = ({row["id"]: row for row in db.open_table(
                "documents").to_arrow().to_pylist()}
                if "documents" in db.list_tables().tables else {})
            enriched: list[Row] = []
            for result in results:
                document = documents.get(result["document_id"], {})
                combined = {
                    **result,
                    "title": document.get("title"),
                    "relative_path": document.get("relative_path"),
                    "collection": document.get("collection", result.get("collection")),
                }
                enriched.append({field: combined.get(field) for field in OUTPUT_FIELDS})
            LOGGER.info("phase=query-complete mode=%s results=%d seconds=%.3f",
                        mode, len(enriched), time.perf_counter() - started)
            return enriched

        candidate_limit = max(20, top_k) if mode == "hybrid" else top_k
        text_results: list[Row] = []
        if mode in {"fts", "hybrid"}:
            lexical_started = time.perf_counter()
            lexical_query = lexicalize_query(query)
            LOGGER.info("phase=lexicalize original=%r lexical=%r seconds=%.3f",
                        query, lexical_query, time.perf_counter() - lexical_started)
            fts_started = time.perf_counter()
            text_results = sorted(
                table.search(lexical_query, query_type="fts").where(where).limit(
                    candidate_limit).to_list(),
                key=lambda row: float(row.get("_score", float("-inf"))), reverse=True,
            )
            text_results = [
                {**row, "_fts_rank": rank, "_fts_score": row.get("_score"),
                 "_vector_rank": None, "_vector_distance": None,
                 "_hybrid_score": None, "_rerank_score": None}
                for rank, row in enumerate(text_results, start=1)
            ]
            LOGGER.info("phase=fts-search-complete candidates=%d seconds=%.3f",
                        len(text_results), time.perf_counter() - fts_started)
            if mode == "fts":
                return enrich(text_results[:top_k])
        embedding_started = time.perf_counter()
        embedding_model = model if model is not None else load_embedding_model(
            self.config.model_path, self.config.gpu_layers, self.config.flash_attn)
        vectors, failures = embed_with_retries(embedding_model, [query])
        if failures or vectors[0] is None:
            raise RuntimeError("failed to embed query")
        LOGGER.info("phase=query-embedding-complete seconds=%.3f",
                    time.perf_counter() - embedding_started)
        vector_started = time.perf_counter()
        vector_results = [
            {**row, "_fts_rank": None, "_fts_score": None, "_vector_rank": rank,
             "_vector_distance": row.get("_distance"), "_hybrid_score": None,
             "_rerank_score": None}
            for rank, row in enumerate(table.search(
                vectors[0], vector_column_name="vector").where(where).limit(
                    candidate_limit).to_list(), start=1)
        ]
        LOGGER.info("phase=vector-search-complete candidates=%d seconds=%.3f",
                    len(vector_results), time.perf_counter() - vector_started)
        if mode == "vector":
            return enrich(vector_results[:top_k])
        fusion_started = time.perf_counter()
        fused: dict[str, Row] = {}
        scores: dict[str, float] = {}
        for weight, result_set, rank_field in (
            (1.0, text_results, "_fts_rank"), (2.0, vector_results, "_vector_rank")
        ):
            for rank, result in enumerate(result_set, start=1):
                key = str(result["id"])
                if key not in fused:
                    fused[key] = {**result}
                else:
                    fused[key].update({name: result.get(name) for name in (
                        rank_field, "_fts_score", "_vector_distance") if result.get(name) is not None})
                scores[key] = scores.get(key, 0.0) + weight / (60 + rank)
        candidates = [{**fused[key], "_hybrid_score": scores[key]}
                      for key in sorted(scores, key=scores.get, reverse=True)]
        LOGGER.info("phase=fusion-complete candidates=%d seconds=%.3f",
                    len(candidates), time.perf_counter() - fusion_started)
        policy = "auto" if rerank is None else "always" if rerank else "never"
        compare = min(2, len(candidates))
        agreed = bool(compare and len(text_results) >= compare and len(vector_results) >= compare
                      and [row["id"] for row in text_results[:compare]]
                      == [row["id"] for row in vector_results[:compare]])
        should_rerank = rerank is True or (rerank is None and bool(candidates) and not agreed)
        reason = ("forced" if rerank is True else "disabled" if rerank is False
                  else "no-candidates" if not candidates
                  else f"top-{compare}-agree" if agreed else f"top-{compare}-disagree")
        LOGGER.info("phase=rerank-decision policy=%s decision=%s reason=%s",
                    policy, "run" if should_rerank else "skip", reason)
        if should_rerank:
            try:
                rerank_limit = min(
                    len(candidates), max(top_k, self.config.rerank_candidates))
                LOGGER.info("phase=rerank-selection fused=%d selected=%d",
                            len(candidates), rerank_limit)
                candidates = candidates[:rerank_limit]
                reranker = reranker or self._reranker
                if reranker is None:
                    load_started = time.perf_counter()
                    reranker = load_reranker_model(
                        self.config.reranker_model_path, self.config.gpu_layers,
                        self.config.flash_attn, self.config.reranker_context)
                    self._reranker = reranker
                    LOGGER.info("phase=load-reranker-complete seconds=%.3f",
                                time.perf_counter() - load_started)
                rerank_started = time.perf_counter()
                rerank_scores = _rerank(
                    reranker, query, candidates, self.config.rerank_max_tokens)
                candidates = [{**candidate, "_rerank_score": score}
                              for candidate, score in zip(candidates, rerank_scores)]
                candidates.sort(key=lambda row: (row["_rerank_score"],
                                                  row["_hybrid_score"]), reverse=True)
                LOGGER.info("phase=rerank-complete candidates=%d seconds=%.3f",
                            len(candidates), time.perf_counter() - rerank_started)
            except Exception as error:
                if "download_models.ps1" in str(error):
                    raise
                raise RuntimeError(
                    f"Reranking failed: {error}. Run "
                    ".\\scripts\\download_models.ps1 -Model reranker"
                ) from error
        return enrich(candidates[:top_k])


def _write_json(value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    sys.stdout.buffer.write(payload.encode("utf-8"))
    sys.stdout.flush()


def _preload_embedding_model(indexer: DocumentIndexer, mode: str) -> Any | None:
    if mode not in {"vector", "hybrid"}:
        return None
    started = time.perf_counter()
    model = load_embedding_model(
        indexer.config.model_path, indexer.config.gpu_layers,
        indexer.config.flash_attn)
    LOGGER.info("phase=load-model-complete seconds=%.3f", time.perf_counter() - started)
    return model


def _run_shell(indexer: DocumentIndexer, args: Any) -> None:
    rerank = True if getattr(args, "always_rerank", False) else False if args.no_rerank else None
    embedding_model = _preload_embedding_model(indexer, args.mode)
    LOGGER.info("phase=shell-ready mode=%s; type exit or quit to stop", args.mode)
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.casefold() in {"exit", "quit"}:
            break
        if not query:
            continue
        try:
            _write_json(indexer.search(
                query, collections=args.collection, mode=args.mode, top_k=args.top_k,
                model=embedding_model, rerank=rerank))
        except (RuntimeError, ValueError) as error:
            clear_progress()
            LOGGER.error("%s", error)


def _run_server(indexer: DocumentIndexer, args: Any) -> None:
    rerank = True if args.always_rerank else False if args.no_rerank else None
    embedding_model = _preload_embedding_model(indexer, args.mode)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/collections":
                self._send(200, {"collections": indexer.list_collections()})
                return
            if self.path != "/health":
                self._send(404, {"error": "not found"})
                return
            self._send(200, {"status": "ok"})

        def do_POST(self) -> None:
            if self.path != "/query":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                query = payload.get("query")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("'query' must be a non-empty string")
                collections = payload.get("collections", args.collection)
                if collections is not None and (
                        not isinstance(collections, list)
                        or not all(isinstance(value, str) for value in collections)):
                    raise ValueError("'collections' must be an array of strings")
                request_rerank = payload.get("rerank", rerank)
                if not isinstance(request_rerank, bool) and request_rerank is not None:
                    raise ValueError("'rerank' must be a boolean or null")
                result = indexer.search(
                    query,
                    collections=collections,
                    mode=payload.get("mode", args.mode),
                    top_k=payload.get("top_k", args.top_k),
                    model=embedding_model,
                    rerank=request_rerank,
                )
                self._send(200, result)
            except (json.JSONDecodeError, TypeError, ValueError, RuntimeError) as error:
                clear_progress()
                self._send(400, {"error": str(error)})

        def log_message(self, format: str, *values: Any) -> None:
            LOGGER.info("http " + format, *values)

    server = HTTPServer((args.host, args.port), Handler)
    LOGGER.info("phase=server-ready address=http://%s:%d; models stay loaded", args.host,
                args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _run_benchmark(indexer: DocumentIndexer, args: Any) -> dict[str, Any]:
    try:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read benchmark cases: {error}") from error
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark cases must be a non-empty JSON array")
    for case in cases:
        if (not isinstance(case, dict) or not str(case.get("query", "")).strip()
                or not isinstance(case.get("expected_paths"), list)
                or not case["expected_paths"]):
            raise ValueError(
                "each benchmark case needs query and non-empty expected_paths")

    rerank = True if getattr(args, "always_rerank", False) else False if args.no_rerank else None
    embedding_model = _preload_embedding_model(indexer, args.mode)
    measured: list[dict[str, Any]] = []
    for number, case in enumerate(cases, start=1):
        started = time.perf_counter()
        results = indexer.search(
            case["query"], collections=args.collection, mode=args.mode,
            top_k=args.top_k, model=embedding_model, rerank=rerank)
        elapsed = time.perf_counter() - started
        expected = set(map(str, case["expected_paths"]))
        paths = [str(result.get("relative_path")) for result in results]
        matched = expected.intersection(paths)
        first_rank = next(
            (rank for rank, path in enumerate(paths, start=1) if path in expected), None)
        measured.append({
            "query": case["query"],
            "seconds": elapsed,
            "recall": len(matched) / len(expected),
            "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
            "matched_paths": sorted(matched),
            "top_paths": paths,
        })
        LOGGER.info("phase=benchmark-case case=%d/%d recall=%.3f rr=%.3f seconds=%.3f",
                    number, len(cases), measured[-1]["recall"],
                    measured[-1]["reciprocal_rank"], elapsed)
    return {
        "cases": measured,
        "summary": {
            "count": len(measured),
            "mean_recall": sum(case["recall"] for case in measured) / len(measured),
            "mean_reciprocal_rank": sum(
                case["reciprocal_rank"] for case in measured) / len(measured),
            "total_seconds": sum(case["seconds"] for case in measured),
        },
    }


def _add_search_arguments(command: Any) -> None:
    command.add_argument("--collection", action="append")
    command.add_argument("--mode", choices=("vector", "fts", "hybrid"), default="hybrid")
    command.add_argument("--top-k", type=int, default=5)
    command.add_argument("--reranker-model", type=Path, default=Path(DEFAULT_RERANKER_MODEL))
    command.add_argument("--rerank-candidates", type=int, default=12)
    command.add_argument("--rerank-max-tokens", type=int, default=1024)
    command.add_argument("--reranker-context", type=int, default=2048)
    rerank = command.add_mutually_exclusive_group()
    rerank.add_argument("--no-rerank", action="store_true")
    rerank.add_argument("--always-rerank", action="store_true")
    command.add_argument("--model", type=Path, default=Path(DEFAULT_EMBEDDING_MODEL))
    command.add_argument("--db-path", type=Path, default=Path("data/lancedb"))
    command.add_argument("--gpu-layers", default="auto")
    command.add_argument("--flash-attn", action="store_true")


def main() -> int:
    parser = argparse.ArgumentParser(description="Index and search documents in LanceDB")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("file", type=Path)
    ingest.add_argument("--collection", default="default")
    query = subparsers.add_parser("query")
    query.add_argument("text")
    _add_search_arguments(query)
    shell = subparsers.add_parser("shell")
    _add_search_arguments(shell)
    server = subparsers.add_parser("serve")
    _add_search_arguments(server)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8181)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("cases", type=Path)
    _add_search_arguments(benchmark)
    reindex = subparsers.add_parser("reindex-fts")
    reindex.add_argument("--db-path", type=Path, default=Path("data/lancedb"))
    ingest.add_argument("--model", type=Path, default=Path(DEFAULT_EMBEDDING_MODEL))
    ingest.add_argument("--db-path", type=Path, default=Path("data/lancedb"))
    ingest.add_argument("--chunk-size", type=int, default=900)
    ingest.add_argument("--chunk-overlap", type=int)
    ingest.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    ingest.add_argument("--gpu-layers", default="auto")
    ingest.add_argument("--flash-attn", action="store_true")
    args = parser.parse_args()
    configure_colored_logging()
    try:
        config = DocumentIndexerConfig(
            model_path=getattr(args, "model", Path(DEFAULT_EMBEDDING_MODEL)),
            reranker_model_path=getattr(
                args, "reranker_model", Path(DEFAULT_RERANKER_MODEL)),
            db_path=args.db_path,
            chunk_size=getattr(args, "chunk_size", 900),
            chunk_overlap=getattr(args, "chunk_overlap", None),
            batch_size=getattr(args, "batch_size", EMBEDDING_BATCH_SIZE),
            gpu_layers=getattr(args, "gpu_layers", "auto"),
            rerank_candidates=getattr(args, "rerank_candidates", 12),
            rerank_max_tokens=getattr(args, "rerank_max_tokens", 1024),
            reranker_context=getattr(args, "reranker_context", 2048),
            flash_attn=getattr(args, "flash_attn", False),
        )
        indexer = DocumentIndexer(config)
        if args.command == "ingest":
            result = indexer.index(args.file, collection=args.collection)
        elif args.command == "reindex-fts":
            result = indexer.reindex_fts()
        elif args.command == "shell":
            _run_shell(indexer, args)
            return 0
        elif args.command == "serve":
            _run_server(indexer, args)
            return 0
        elif args.command == "benchmark":
            result = _run_benchmark(indexer, args)
        else:
            result = indexer.search(
                args.text, collections=args.collection, mode=args.mode, top_k=args.top_k,
                rerank=True if args.always_rerank else False if args.no_rerank else None)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        clear_progress()
        LOGGER.error("%s", error)
        return 1
    _write_json(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
