"""Small shared runtime used by the document indexer."""

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, cast

import pyarrow as pa  # type: ignore[reportMissingTypeStubs]
from llama_cpp import Llama, llama_supports_gpu_offload

DEFAULT_EMBEDDING_MODEL = "models/Qwen3-Embedding-0.6B-Q8_0.gguf"
DEFAULT_RERANKER_MODEL = "models/qwen3-reranker-0.6b-q8_0.gguf"
EMBEDDING_BATCH_SIZE = 32
DOCLING_DOCUMENT_SUFFIXES = {
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods", ".odp",
}
DOCUMENT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".pdf"} | DOCLING_DOCUMENT_SUFFIXES

pa = cast(Any, pa)


@dataclass(frozen=True)
class InputDocument:
    text: str
    collection_root: str
    relative_path: str
    content_hash: str

    @property
    def id(self) -> str:
        return hashlib.sha256(
            f"{self.collection_root}\0{self.relative_path}".encode()
        ).hexdigest()


def hash_value(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(vector: Iterable[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm or not math.isfinite(norm):
        raise RuntimeError("The model returned a zero or non-finite vector")
    result = [float(value / norm) for value in vector]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError("The model returned a non-finite vector")
    return result


def _embed(model: Any, text: str) -> list[float]:
    try:
        response: Any = model.create_embedding(text)
        return _normalize(cast(list[float], response["data"][0]["embedding"]))
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("The model returned an invalid embedding response") from error


def embed_with_retries(model: Any, texts: list[str]) -> tuple[list[list[float] | None], int]:
    try:
        response: Any = model.create_embedding(texts)
        vectors = [_normalize(cast(list[float], item["embedding"]))
                    for item in response["data"]]
        if len(vectors) != len(texts):
            raise RuntimeError("The model returned the wrong number of embeddings")
        return vectors, 0
    except Exception:
        vectors: list[list[float] | None] = []
        failures = 0
        for text in texts:
            for attempt in range(3):
                try:
                    vectors.append(_embed(model, text))
                    break
                except Exception:
                    if attempt == 2:
                        vectors.append(None)
                        failures += 1
        return vectors, failures


def _normalized_root(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").rstrip("/").casefold()


def _input_document(path: Path, root: Path) -> InputDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        # ponytail: text PDFs only; add OCR when scanned PDFs must be supported.
        import pypdfium2 as pdfium
        with pdfium.PdfDocument(path) as pdf:
            text = "\n\n".join(
                pdf[index].get_textpage().get_text_range()
                for index in range(len(pdf))
            )
    elif suffix in DOCLING_DOCUMENT_SUFFIXES:
        from docling.document_converter import DocumentConverter
        text = DocumentConverter().convert(path).document.export_to_markdown()
    else:
        text = path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufffe", "")
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return InputDocument(text, _normalized_root(root), relative,
                         hashlib.sha256(text.encode()).hexdigest())


def read_input_documents(paths: list[Path]) -> list[InputDocument]:
    result: list[InputDocument] = []
    for path in paths:
        resolved = path.resolve()
        if path.is_dir():
            result.extend(
                _input_document(file, resolved)
                for file in sorted(resolved.rglob("*"))
                if file.is_file() and file.suffix.lower() in DOCUMENT_SUFFIXES
            )
        elif path.is_file():
            result.append(_input_document(resolved, resolved.parent))
        else:
            raise FileNotFoundError(f"Document path not found: {path}")
    return result


def documents_schema() -> Any:
    return pa.schema([
        pa.field("id", pa.string()), pa.field("collection_root", pa.string()),
        pa.field("collection", pa.string()),
        pa.field("relative_path", pa.string()), pa.field("content_hash", pa.string()),
        pa.field("embedding_fingerprint", pa.string()),
        pa.field("title", pa.string()), pa.field("total_chunks", pa.int32()),
        pa.field("active", pa.bool_()), pa.field("updated_at", pa.string()),
    ])


def chunks_schema(dimension: int) -> Any:
    return pa.schema([
        pa.field("id", pa.string()), pa.field("document_id", pa.string()),
        pa.field("collection", pa.string()),
        pa.field("content_hash", pa.string()),
        pa.field("embedding_fingerprint", pa.string()),
        pa.field("chunk_index", pa.int32()), pa.field("total_chunks", pa.int32()),
        pa.field("position", pa.int64()), pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dimension)),
        pa.field("active", pa.bool_()),
    ])


def rows(table: Any, where: str | None = None) -> list[dict[str, Any]]:
    if where is None:
        return table.to_arrow().to_pylist()
    count = table.count_rows(where)
    return table.search().where(where).limit(count).to_list() if count else []


def upsert(table: Any, values: list[dict[str, Any]]) -> None:
    if values:
        table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(values)


def _cpu_thread_budget() -> int:
    available = os.cpu_count() or 1
    return max(1, min(8, available - max(1, available // 3)))


def load_model(path: str, *, embedding: bool = False,
               gpu_layers: str | int = "auto",
               threads: int | None = None,
               pooling_type: int = -1,
               context: int | None = None,
               flash_attn: bool = False) -> Any:
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    layers = (-1 if llama_supports_gpu_offload() else 0) \
        if gpu_layers == "auto" else int(gpu_layers)
    context = context or (2048 if "gemma" in model_path.name.lower() else 4096)
    threads = threads or _cpu_thread_budget()
    return Llama(model_path=str(model_path), embedding=embedding, n_ctx=context,
                 n_batch=context, n_ubatch=context, n_threads=threads,
                 n_threads_batch=threads, n_gpu_layers=layers,
                 pooling_type=pooling_type, flash_attn=flash_attn, verbose=False)
