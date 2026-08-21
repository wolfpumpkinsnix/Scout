"""Document chunking, embedding, and LanceDB persistence."""
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportCallIssue=false, reportArgumentType=false

import argparse
import bisect
import fnmatch
import json
import logging
import math
import re
import sys
import tempfile
import time
from collections import Counter
from email import policy
from email.parser import BytesParser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast
from urllib.parse import parse_qs, unquote, urlsplit

import lancedb
from lancedb.index import FTS
from llama_cpp import LLAMA_POOLING_TYPE_RANK
from src.indexer_support import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    DOCUMENT_SUFFIXES,
    EMBEDDING_BATCH_SIZE,
    DOCLING_DOCUMENT_SUFFIXES,
    TEXT_FTS_MIGRATION_SQL,
    InputDocument,
    chunks_schema,
    documents_schema,
    embed_with_retries,
    hash_value,
    now,
    rows,
    upsert,
    load_model,
    normalize_tech_tokens,
    read_input_documents,
)
from src.logging_utils import clear_progress, configure_colored_logging, update_progress

LOGGER = logging.getLogger("document-indexer")


@dataclass(frozen=True)
class DocumentIndexerConfig:
    model_path: Path = Path(DEFAULT_EMBEDDING_MODEL)
    db_path: Path = Path("data/lancedb")
    index_path: Path = Path("index.yml")
    chunk_size: int = 900
    chunk_overlap: int | None = None
    batch_size: int = EMBEDDING_BATCH_SIZE
    gpu_layers: str | int = "auto"
    reranker_model_path: Path = Path(DEFAULT_RERANKER_MODEL)
    rerank_candidates: int = 12
    rerank_max_tokens: int = 1024
    reranker_context: int = 2048
    flash_attn: bool = False
    min_score: float = 0.0


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
CHUNKER_VERSION = "adaptive-overlap-v3"
LARGE_SECTION_CHARS = 1_000_000
ARTICLE_HEADING = re.compile(r"(?m)^[ \t]*(Article[ \t]+\d+[A-Za-z]?)[ \t]*$")
OUTPUT_FIELDS = (
    "id", "document_id", "collection", "title", "relative_path", "chunk_index",
    "total_chunks", "position", "text", "_fts_rank", "_fts_score", "_vector_rank",
    "_vector_distance", "_hybrid_score", "_rerank_score",
)


def _read_collection_index(path: Path) -> dict[str, Row]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read collection index {path}: {error}") from error
    collections = payload.get("collections") if isinstance(payload, dict) else None
    if not isinstance(collections, dict):
        raise ValueError(f"Invalid collection index {path}")
    result: dict[str, Row] = {}
    for name, entry in collections.items():
        if (isinstance(name, str) and isinstance(entry, dict)
                and isinstance(entry.get("path"), str)
                and isinstance(entry.get("pattern"), str)):
            result[name] = {
                "name": name, "path": entry["path"], "pattern": entry["pattern"],
            }
    if len(result) != len(collections):
        raise ValueError(f"Invalid collection entry in {path}")
    return result


def _write_collection_index(path: Path, collections: dict[str, Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"collections": {name: {
        "path": entry["path"], "pattern": entry["pattern"],
    } for name, entry in sorted(collections.items())}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def lexicalize_query(query: str) -> str:
    tokens = [token.text for token in lancedb.tokenize(
        normalize_tech_tokens(query), **FTS_OPTIONS)]
    lexical = " ".join(token for token in tokens if token.casefold() not in LEXICAL_STOP_WORDS)
    return lexical or query


def _dedup_by_document(results: list[Row]) -> list[Row]:
    seen: set[str] = set()
    return [row for row in results
            if not (str(row["document_id"]) in seen or seen.add(str(row["document_id"])))]


_TOC_LINE = re.compile(r"^\s*(.{0,100}?\.{4,}\s*\d+|\d{1,4})\s*$")


def _is_boilerplate(text: str) -> bool:
    # ponytail: TOC heuristic only — dot-leader/page-number lines.
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    toc_lines = sum(1 for line in lines if _TOC_LINE.match(line))
    return toc_lines / len(lines) > 0.3


def _create_fts(table: Any, *, rebuild: bool = False) -> None:
    has_fts = any(index.index_type == "FTS" for index in table.list_indices())
    if "text_fts" not in table.schema.names:
        table.add_columns({"text_fts": TEXT_FTS_MIGRATION_SQL})
        has_fts = False  # old index was on "text"; force rebuild on text_fts
    if rebuild or not has_fts:
        table.create_index(
            "text_fts", config=FTS(with_position=True, **FTS_OPTIONS), replace=True)
    else:
        table.optimize()


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


def _whitespace_map(text: str, start: int, end: int) -> tuple[str, list[int]]:
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
    return "".join(collapsed), offsets


def _whitespace_position(needle: str, normalized: tuple[str, list[int]],
                         start: int, end: int) -> int:
    text, offsets = normalized
    match = text.find(" ".join(needle.split()), bisect.bisect_left(offsets, start),
                      bisect.bisect_left(offsets, end))
    return offsets[match] if match >= 0 and offsets else -1


def _chunk_position(text: str, needle: str, start: int, end: int,
                    fallback: int, path: str,
                    normalized: tuple[str, list[int]] | None = None) -> int:
    normalized = normalized or _whitespace_map(text, start, end)
    position = text.find(needle, start, end)
    if position < 0:
        position = _whitespace_position(needle, normalized, start, end)
    if position < 0:
        anchor = " ".join(needle.split()[:12])
        position = _whitespace_position(anchor, normalized, start, end)
        if position >= 0:
            LOGGER.info("phase=chunk-position-anchor document=%s position=%d", path, position)
    if position < 0:
        LOGGER.warning("phase=chunk-position-fallback document=%s section=%d", path, fallback)
        return fallback
    return position


def _overlap_suffix(text: str, tokenizer: Any, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    try:
        encoded = tokenizer.get_tokenizer()(
            text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        if len(offsets) <= max_tokens:
            return text
        minimum_start = offsets[-max_tokens][0]
        for pattern in (r"\n+|(?<=[.!?])\s+", r"\s+"):
            match = re.compile(pattern).search(text, max(0, minimum_start - 1))
            if match:
                boundary = match.end()
                if boundary >= minimum_start:
                    return text[boundary:]
        return text[minimum_start:]
    except (AttributeError, KeyError, TypeError):
        pass
    if tokenizer.count_tokens(text) <= max_tokens:
        return text
    for pattern in (r"\n+|(?<=[.!?])\s+", r"\s+"):
        boundaries = [match.end() for match in re.finditer(pattern, text)]
        low, high, answer = 0, len(boundaries) - 1, None
        while low <= high:
            middle = (low + high) // 2
            candidate = text[boundaries[middle]:]
            if tokenizer.count_tokens(candidate) <= max_tokens:
                answer, high = candidate, middle - 1
            else:
                low = middle + 1
        if answer:
            return answer
    return ""


def _pdf_text(path: Path) -> str:
    # ponytail: text PDFs only; add OCR when scanned PDFs must be supported.
    import pypdfium2 as pdfium
    with pdfium.PdfDocument(path) as pdf:
        return "\n\n".join(
            pdf[index].get_textpage().get_text_range()
            for index in range(len(pdf))
        ).replace("\r\n", "\n").replace("\r", "\n").replace("\ufffe", "")


@lru_cache(maxsize=8)
def _docling_chunkers(max_tokens: int) -> tuple[Any, Any]:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

    model = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        tokenizer = HuggingFaceTokenizer.from_pretrained(
            model, max_tokens=max_tokens, local_files_only=True)
    except OSError:
        tokenizer = HuggingFaceTokenizer.from_pretrained(model, max_tokens=max_tokens)
    hybrid = HybridChunker(tokenizer=tokenizer)
    return hybrid, LineBasedTokenChunker(tokenizer=hybrid.tokenizer, prefix="")


def chunk_documents(
    documents: list[InputDocument], chunk_size: int, overlap_tokens: int
) -> list[DocumentChunk]:
    from dataclasses import replace
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    hybrid_chunker, line_chunker = _docling_chunkers(chunk_size - overlap_tokens)
    docling_tokenizer: Any = hybrid_chunker.tokenizer
    chunks: list[DocumentChunk] = []
    update_progress("Chunking", 0, len(documents))
    for document_number, document in enumerate(documents, start=1):
        source = Path(document.collection_root) / document.relative_path
        suffix = source.suffix.lower()
        converted_document = None
        if suffix == ".pdf":
            source_text = document.text or _pdf_text(source)
        elif suffix in DOCLING_DOCUMENT_SUFFIXES:
            converted_document = converter.convert(source).document
            source_text = converted_document.export_to_markdown()
            source_text = source_text.replace("\r\n", "\n").replace(
                "\r", "\n").replace("\ufffe", "")
        else:
            source_text = document.text
            converted_document = converter.convert_string(
                source_text, InputFormat.MD, name=source.stem).document
        working_document = replace(document, text=source_text)
        sections = (_pdf_sections(source_text) if suffix == ".pdf"
                    else [(0, len(source_text), "")])
        document_chunks: list[Any] = []
        for section_start, section_end, markdown in sections:
            converted = (converter.convert_string(
                markdown, InputFormat.MD, name=source.stem).document
                if suffix == ".pdf" else converted_document)
            section_length = len(markdown or source_text)
            chunker = line_chunker if section_length >= LARGE_SECTION_CHARS else hybrid_chunker
            LOGGER.info("phase=chunker-select document=%s chars=%d chunker=%s",
                        document.relative_path, section_length,
                        "line" if section_length >= LARGE_SECTION_CHARS else "hybrid")
            section_chunks = list(chunker.chunk(converted))
            normalized = _whitespace_map(source_text, section_start, section_end)
            previous_text = ""
            previous_position = section_start
            for chunk in section_chunks:
                if _is_boilerplate(chunk.text):
                    continue
                search_start = section_start if not previous_text else previous_position + 1
                chunk_position = _chunk_position(
                    source_text, chunk.text, search_start, section_end,
                    search_start, document.relative_path, normalized)
                overlap = _overlap_suffix(previous_text, docling_tokenizer, overlap_tokens)
                text = chunker.contextualize(chunk)
                stored_position = chunk_position
                if overlap:
                    stored_position = _chunk_position(
                        source_text, overlap, previous_position, chunk_position,
                        chunk_position, document.relative_path, normalized)
                    text = f"{text[:-len(chunk.text)]}{overlap}\n{chunk.text}"
                chunks.append(DocumentChunk(
                    working_document, len(document_chunks), text, stored_position))
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
        if not 0 <= config.min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        self.config = config
        self._reranker: Any | None = None

    def index(
        self, path: Path, model: Any | None = None, collection: str = "default",
        pattern: str | None = None,
    ) -> dict[str, int | str]:
        started = time.perf_counter()
        if not collection.strip():
            raise ValueError("collection must not be empty")
        if not path.exists():
            raise FileNotFoundError(f"Document path not found: {path}")
        LOGGER.info("phase=scan path=%s", path)
        documents = read_input_documents(path)
        if pattern is not None:
            documents = [
                document for document in documents
                if fnmatch.fnmatchcase(document.relative_path, pattern)
                or (pattern.startswith("**/")
                    and fnmatch.fnmatchcase(document.relative_path, pattern[3:]))
            ]
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
        documents_table.delete("active = false")
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
        if chunks_table is not None:
            chunks_table.delete("active = false")
        active_chunk_counts: Counter[str] = Counter()
        if chunks_table is not None and document_ids:
            ids = ",".join(repr(value) for value in document_ids.values())
            where = f"active = true AND document_id IN ({ids})"
            count = chunks_table.count_rows(where)
            if count:
                active_chunk_counts.update(
                    row["document_id"] for row in chunks_table.search().where(
                        where).select(["document_id"]).limit(count).to_list()
                )
        pending_documents: list[InputDocument] = []
        for document in documents:
            existing = existing_documents.get(document_ids[document.id])
            active_chunks = active_chunk_counts[document_ids[document.id]]
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
            if deleted_rows:
                ids = ",".join(repr(row["id"]) for row in deleted_rows)
                if chunks_table is not None:
                    chunks_table.delete(f"document_id IN ({ids})")
                documents_table.delete(f"id IN ({ids})")

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
        if "text_fts" not in chunks_table.schema.names:
            chunks_table.add_columns({"text_fts": TEXT_FTS_MIGRATION_SQL})
        LOGGER.info("phase=database-ready path=%s dimension=%d", self.config.db_path, dimension)
        chunk_started = time.perf_counter()
        chunks = chunk_documents(pending_documents, chunk_size, chunk_overlap)
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
                    "text_fts": normalize_tech_tokens(chunk.text),
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
        obsolete_chunk_ids: list[str] = []
        document_rows_by_id: dict[str, Row] = {
            document.id: row for document, row in zip(pending_documents, document_rows)
        }
        for document in pending_documents:
            document_chunk_rows = rows_by_document[document.id]
            expected = total_chunks_by_document[document.id]
            if len(document_chunk_rows) == expected and expected:
                active_documents.append({**document_rows_by_id[document.id], "active": True})
                active_chunks.extend([{**row, "active": True} for row in document_chunk_rows])
                obsolete_chunk_ids.extend(
                    str(row["id"]) for row in old_chunks_by_document[document.id])
        upsert(chunks_table, active_chunks)
        upsert(documents_table, active_documents)
        if obsolete_chunk_ids:
            ids = ",".join(repr(value) for value in obsolete_chunk_ids)
            chunks_table.delete(f"id IN ({ids})")
        pending_ids = ",".join(
            repr(document_ids[document.id]) for document in pending_documents)
        chunks_table.delete(f"active = false AND document_id IN ({pending_ids})")
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
        _create_fts(table, rebuild=True)
        result: dict[str, int | str] = {
            "db_path": str(self.config.db_path),
            "indexed_chunks": table.count_rows(),
        }
        LOGGER.info("phase=fts-reindex-complete chunks=%d seconds=%.3f",
                    result["indexed_chunks"], time.perf_counter() - started)
        return result

    def collections(self) -> list[Row]:
        return list(_read_collection_index(self.config.index_path).values())

    def add_collection(self, name: str, path: Path,
                       pattern: str = "**/*.md") -> Row:
        name = name.strip()
        if not name:
            name = path.resolve().name or "root"
        if not path.is_dir():
            raise ValueError(f"collection path must be a directory: {path}")
        if not pattern.strip():
            raise ValueError("collection pattern must not be empty")
        collections = _read_collection_index(self.config.index_path)
        if name in collections:
            raise ValueError(f"collection already exists: {name}")
        resolved_path = str(path.resolve())
        if any(entry["path"] == resolved_path and entry["pattern"] == pattern
               for entry in collections.values()):
            raise ValueError(f"collection already exists for path and pattern: {resolved_path}")
        entry = {"name": name, "path": resolved_path, "pattern": pattern}
        collections[name] = entry
        _write_collection_index(self.config.index_path, collections)
        return entry

    def remove_document(self, document_id: str) -> int:
        db: Any = lancedb.connect(str(self.config.db_path))
        if "documents" not in db.list_tables().tables:
            return 0
        documents_table = db.open_table("documents")
        documents = rows(
            documents_table, f"id = {document_id!r} AND active = true")
        if not documents:
            return 0
        if "chunks" in db.list_tables().tables:
            chunks_table = db.open_table("chunks")
            chunks_table.delete(f"document_id = {document_id!r}")
            _create_fts(chunks_table)
        documents_table.delete(f"id = {document_id!r}")
        return 1

    def remove_collection(self, name: str) -> int:
        collections = _read_collection_index(self.config.index_path)
        if name not in collections:
            raise ValueError(f"collection not found: {name}")
        del collections[name]
        _write_collection_index(self.config.index_path, collections)
        db: Any = lancedb.connect(str(self.config.db_path))
        if "documents" not in db.list_tables().tables:
            return 0
        documents_table = db.open_table("documents")
        documents = rows(
            documents_table, f"collection = {name!r} AND active = true")
        if "chunks" in db.list_tables().tables:
            chunks_table = db.open_table("chunks")
            chunks_table.delete(f"collection = {name!r}")
            _create_fts(chunks_table)
        documents_table.delete(f"collection = {name!r}")
        return len(documents)

    def update_collections(self, names: list[str] | None = None,
                           model: Any | None = None) -> list[dict[str, Any]]:
        collections = _read_collection_index(self.config.index_path)
        selected = names or sorted(collections)
        results = []
        for name in selected:
            entry = collections.get(name)
            if entry is None:
                raise ValueError(f"collection not found: {name}")
            results.append(self.index(
                Path(entry["path"]), model=model, collection=name,
                pattern=entry["pattern"],
            ))
        return results

    def list_collections(self) -> list[str]:
        db: Any = lancedb.connect(str(self.config.db_path))
        if "documents" not in db.list_tables().tables:
            return []
        return sorted({
            str(row["collection"])
            for row in db.open_table("documents").to_arrow().to_pylist()
            if row.get("active") and row.get("collection")
        })

    def list_documents(self, collection: str | None = None) -> list[Row]:
        db: Any = lancedb.connect(str(self.config.db_path))
        if "documents" not in db.list_tables().tables:
            return []
        documents = [
            {
                "id": row["id"],
                "collection": row["collection"],
                "relative_path": row["relative_path"],
                "title": row["title"],
                "total_chunks": row["total_chunks"],
                "updated_at": row["updated_at"],
            }
            for row in db.open_table("documents").to_arrow().to_pylist()
            if row.get("active") and (collection is None or row.get("collection") == collection)
        ]
        return sorted(documents, key=lambda row: (row["collection"], row["relative_path"]))

    def get_document(self, document_id: str) -> Row | None:
        db: Any = lancedb.connect(str(self.config.db_path))
        if "documents" not in db.list_tables().tables:
            return None
        document = next(
            (row for row in db.open_table("documents").to_arrow().to_pylist()
             if row.get("id") == document_id and row.get("active")),
            None,
        )
        if document is None:
            return None
        chunks = []
        if "chunks" in db.list_tables().tables:
            chunks = [
                row for row in db.open_table("chunks").to_arrow().to_pylist()
                if row.get("document_id") == document_id and row.get("active")
            ]
        chunks.sort(key=lambda row: int(row["chunk_index"]))
        return {
            "id": document["id"],
            "collection": document["collection"],
            "relative_path": document["relative_path"],
            "title": document["title"],
            "total_chunks": document["total_chunks"],
            "updated_at": document["updated_at"],
            "text": "\n\n".join(str(row["text"]) for row in chunks),
        }

    def status(self) -> dict[str, Any]:
        db: Any = lancedb.connect(str(self.config.db_path))
        documents = self.list_documents()
        chunks = 0
        if "chunks" in db.list_tables().tables:
            chunks = db.open_table("chunks").count_rows("active = true")
        return {
            "documents": len(documents),
            "chunks": chunks,
            "collections": self.list_collections(),
            "db_path": str(self.config.db_path),
        }

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
            document_ids = sorted({str(row["document_id"]) for row in results})
            documents: dict[str, Row] = {}
            if document_ids and "documents" in db.list_tables().tables:
                values = ",".join(repr(value) for value in document_ids)
                documents = {row["id"]: row for row in rows(
                    db.open_table("documents"), f"id IN ({values})")}
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
            text_results = _dedup_by_document(text_results)
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
        vector_results = _dedup_by_document(vector_results)
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
        candidates = _dedup_by_document(
            [{**fused[key], "_hybrid_score": scores[key]}
             for key in sorted(scores, key=scores.get, reverse=True)])
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
                if self.config.min_score > 0:
                    candidates = [row for row in candidates
                                  if row["_rerank_score"] >= self.config.min_score]
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
        def _json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _multipart_body(self) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            headers = (
                f"Content-Type: {self.headers['Content-Type']}\r\n"
                "MIME-Version: 1.0\r\n\r\n"
            ).encode("ascii")
            message = BytesParser(policy=policy.default).parsebytes(headers + body)
            if not message.is_multipart():
                raise ValueError("invalid multipart/form-data body")
            fields: dict[str, str] = {}
            files: list[tuple[str, bytes]] = []
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                filename = part.get_filename()
                if not isinstance(name, str):
                    continue
                if filename is None:
                    fields[name] = part.get_content()
                else:
                    files.append((Path(filename).name, part.get_payload(decode=True) or b""))
            return fields, files

        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == "/collections":
                self._send(200, {"collections": indexer.collections()})
                return
            if parsed.path.startswith("/collections/"):
                name = unquote(parsed.path[len("/collections/"):])
                collection = next(
                    (entry for entry in indexer.collections() if entry["name"] == name), None)
                if collection is None:
                    self._send(404, {"error": "collection not found"})
                else:
                    self._send(200, collection)
                return
            if parsed.path == "/documents":
                collection = parse_qs(parsed.query).get("collection", [None])[0]
                self._send(200, {"documents": indexer.list_documents(collection)})
                return
            if parsed.path.startswith("/documents/"):
                document = indexer.get_document(unquote(parsed.path[len("/documents/"):]))
                if document is None:
                    self._send(404, {"error": "document not found"})
                else:
                    self._send(200, document)
                return
            if parsed.path == "/status":
                self._send(200, {
                    **indexer.status(),
                    "embedding_model_loaded": embedding_model is not None,
                    "reranker_model_loaded": indexer._reranker is not None,
                })
                return
            if parsed.path != "/health":
                self._send(404, {"error": "not found"})
                return
            self._send(200, {
                "status": "ok",
                "embedding_model_loaded": embedding_model is not None,
                "reranker_model_loaded": indexer._reranker is not None,
            })

        def do_DELETE(self) -> None:
            try:
                path = urlsplit(self.path).path
                if path.startswith("/documents/"):
                    document_id = unquote(path[len("/documents/"):])
                    if not indexer.remove_document(document_id):
                        self._send(404, {"error": "document not found"})
                    else:
                        self._send(200, {"deleted": True, "document_id": document_id})
                    return
                if path.startswith("/collections/"):
                    name = unquote(path[len("/collections/"):])
                    deleted = indexer.remove_collection(name)
                    self._send(200, {"deleted": True, "collection": name,
                                     "documents": deleted})
                    return
                self._send(404, {"error": "not found"})
            except (OSError, ValueError, RuntimeError) as error:
                self._send(400, {"error": str(error)})

        def do_POST(self) -> None:
            try:
                path = urlsplit(self.path).path
                if path == "/collections":
                    payload = self._json_body()
                    name = payload.get("name")
                    collection_path = payload.get("path")
                    pattern = payload.get("pattern", "**/*.md")
                    if name is not None and not isinstance(name, str):
                        raise ValueError("'name' must be a string")
                    if not isinstance(collection_path, str):
                        raise ValueError("'path' must be a string")
                    if not isinstance(pattern, str):
                        raise ValueError("'pattern' must be a string")
                    entry = indexer.add_collection(
                        name or "", Path(collection_path), pattern)
                    self._send(201, {
                        "collection": entry,
                        "results": indexer.update_collections(
                            [entry["name"]], embedding_model),
                    })
                    return
                if path == "/update":
                    payload = self._json_body()
                    names = payload.get("collections")
                    if names is not None and (
                            not isinstance(names, list)
                            or not all(isinstance(value, str) for value in names)):
                        raise ValueError("'collections' must be an array of strings")
                    self._send(200, {"results": indexer.update_collections(names, embedding_model)})
                    return
                if path == "/feedback":
                    payload = self._json_body()
                    if not isinstance(payload.get("document_id"), str):
                        raise ValueError("'document_id' must be a string")
                    if not isinstance(payload.get("relevant"), bool):
                        raise ValueError("'relevant' must be a boolean")
                    LOGGER.info("feedback document_id=%s relevant=%s query=%r",
                                payload["document_id"], payload["relevant"],
                                payload.get("query"))
                    self._send(202, {"accepted": True})
                    return
                if path in {"/ingest", "/documents"}:
                    content_type = self.headers.get("Content-Type", "")
                    if path == "/documents" and content_type.startswith("application/json"):
                        payload = self._json_body()
                        source_path = payload.get("path")
                        collection = payload.get("collection", "default")
                        if not isinstance(source_path, str) or not source_path.strip():
                            raise ValueError("'path' must be a non-empty string")
                        if not isinstance(collection, str) or not collection.strip():
                            raise ValueError("'collection' must be a non-empty string")
                        result = indexer.index(
                            Path(source_path), model=embedding_model, collection=collection)
                        self._send(201, result)
                        return
                    if path == "/ingest" and content_type.startswith("application/json"):
                        payload = self._json_body()
                        names = payload.get("collections")
                        if names is not None and (
                                not isinstance(names, list)
                                or not all(isinstance(value, str) for value in names)):
                            raise ValueError("'collections' must be an array of strings")
                        self._send(200, {
                            "results": indexer.update_collections(names, embedding_model)
                        })
                        return
                    if not content_type.startswith("multipart/form-data"):
                        raise ValueError("Content-Type must be multipart/form-data")
                    fields, uploads = self._multipart_body()
                    uploads = [upload for upload in uploads if upload[0]]
                    collection = fields.get("collection", "default").strip()
                    if not collection:
                        raise ValueError("'collection' must not be empty")
                    if not uploads:
                        raise ValueError("at least one file is required in 'files'")
                    results = []
                    with tempfile.TemporaryDirectory() as directory:
                        for index, (filename, content) in enumerate(uploads):
                            if Path(filename).suffix.lower() not in DOCUMENT_SUFFIXES:
                                raise ValueError(f"unsupported file: {filename}")
                            target = Path(directory) / f"{index}-{filename}"
                            target.write_bytes(content)
                            results.append(indexer.index(
                                target, model=embedding_model, collection=collection))
                    self._send(200, {"files": len(results), "results": results})
                    return
                if path != "/query":
                    self._send(404, {"error": "not found"})
                    return
                payload = self._json_body()
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
            except (json.JSONDecodeError, OSError, TypeError, ValueError, RuntimeError) as error:
                clear_progress()
                self._send(400, {"error": str(error)})

        def do_PUT(self) -> None:
            try:
                path = urlsplit(self.path).path
                if not path.startswith("/collections/"):
                    self._send(404, {"error": "not found"})
                    return
                name = unquote(path[len("/collections/"):])
                payload = self._json_body()
                collection_path = payload.get("path")
                pattern = payload.get("pattern", "**/*.md")
                if not isinstance(collection_path, str) or not isinstance(pattern, str):
                    raise ValueError("'path' and 'pattern' must be strings")
                collections = _read_collection_index(indexer.config.index_path)
                if name not in collections:
                    self._send(404, {"error": "collection not found"})
                    return
                if not Path(collection_path).is_dir():
                    raise ValueError(f"collection path must be a directory: {collection_path}")
                collections[name] = {
                    "name": name, "path": str(Path(collection_path).resolve()),
                    "pattern": pattern,
                }
                _write_collection_index(indexer.config.index_path, collections)
                self._send(200, collections[name])
            except (json.JSONDecodeError, OSError, TypeError, ValueError, RuntimeError) as error:
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
    command.add_argument("--index-path", type=Path, default=Path("index.yml"))
    command.add_argument("--gpu-layers", default="auto")
    command.add_argument("--flash-attn", action="store_true")
    command.add_argument("--min-score", type=float, default=0.0,
                         help="drop reranked results below this score (0 disables)")


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
    update = subparsers.add_parser("update")
    _add_search_arguments(update)
    collection = subparsers.add_parser("collection")
    collection_commands = collection.add_subparsers(dest="collection_command", required=True)
    collection_add = collection_commands.add_parser("add")
    collection_add.add_argument("path", type=Path)
    collection_add.add_argument("--name")
    collection_add.add_argument("--pattern", default="**/*.md")
    collection_add.add_argument("--index-path", type=Path, default=Path("index.yml"))
    collection_delete = collection_commands.add_parser("delete")
    collection_delete.add_argument("name")
    collection_delete.add_argument("--db-path", type=Path, default=Path("data/lancedb"))
    collection_delete.add_argument("--index-path", type=Path, default=Path("index.yml"))
    collection_list = collection_commands.add_parser("list")
    collection_list.add_argument("--index-path", type=Path, default=Path("index.yml"))
    document = subparsers.add_parser("document")
    document_commands = document.add_subparsers(dest="document_command", required=True)
    document_add = document_commands.add_parser("add")
    document_add.add_argument("file", type=Path)
    document_add.add_argument("--collection", default="default")
    document_add.add_argument("--model", type=Path, default=Path(DEFAULT_EMBEDDING_MODEL))
    document_add.add_argument("--db-path", type=Path, default=Path("data/lancedb"))
    document_delete = document_commands.add_parser("delete")
    document_delete.add_argument("id")
    document_delete.add_argument("--db-path", type=Path, default=Path("data/lancedb"))
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
            index_path=getattr(args, "index_path", Path("index.yml")),
            min_score=getattr(args, "min_score", 0.0),
        )
        indexer = DocumentIndexer(config)
        if args.command == "ingest":
            result = indexer.index(args.file, collection=args.collection)
        elif args.command == "collection":
            if args.collection_command == "add":
                entry = indexer.add_collection(args.name or "", args.path, args.pattern)
                result = {
                    "collection": entry,
                    "results": indexer.update_collections(
                        [entry["name"]], _preload_embedding_model(indexer, "vector")),
                }
            elif args.collection_command == "delete":
                result = {"collection": args.name,
                          "documents": indexer.remove_collection(args.name)}
            else:
                result = {"collections": indexer.collections()}
        elif args.command == "document":
            if args.document_command == "add":
                result = indexer.index(args.file, collection=args.collection)
            else:
                result = {"document_id": args.id,
                          "deleted": bool(indexer.remove_document(args.id))}
        elif args.command == "update":
            result = {"results": indexer.update_collections(
                args.collection, _preload_embedding_model(indexer, args.mode))}
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
