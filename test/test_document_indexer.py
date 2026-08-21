# pyright: reportPrivateUsage=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false, reportIncompatibleMethodOverride=false, reportMissingTypeStubs=false
# Test doubles (fakes/mocks) are intentionally untyped; they exercise privates.
import io
import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import lancedb

from src.document_indexer import (
    OUTPUT_FIELDS,
    DocumentChunk,
    DocumentIndexer,
    DocumentIndexerConfig,
    _chunk_position,
    _create_fts,
    _rerank,
    _run_benchmark,
    _run_shell,
    _overlap_suffix,
    chunk_documents,
    lexicalize_query,
)
from src.indexer_support import InputDocument, chunks_schema, read_input_documents
from src.logging_utils import ColoredFormatter, InteractiveConsoleHandler


class FakeEmbeddingModel:
    def n_embd(self):
        return 2

    def tokenize(self, value, **_):
        return value.decode("utf-8").split()

    def detokenize(self, tokens):
        return " ".join(tokens).encode()

    def create_embedding(self, value):
        texts = value if isinstance(value, list) else [value]
        return {"data": [{"embedding": [1.0, float(len(text))]} for text in texts]}


class FakeReranker:
    def __init__(self, outputs=None, context=4096, batch=None):
        self.outputs = outputs
        self.context = context
        self.n_batch = batch or context
        self.prompts = []
        self.batches = []
        self.offset = 0

    def n_ctx(self):
        return self.context

    def tokenize(self, value, **_):
        return list(value)

    def detokenize(self, tokens):
        return bytes(tokens)

    def embed(self, prompts, **_):
        self.prompts.extend(prompts)
        self.batches.append(len(prompts))
        if self.outputs is None:
            return [[0.8, 0.2] for _ in prompts]
        result = self.outputs[self.offset:self.offset + len(prompts)]
        self.offset += len(prompts)
        return result


class FakeChatModel:
    def __init__(self, context=4096, text="Declare it with a type and a name [1]."):
        self.context = context
        self.text = text
        self.prompts = []
        self.options = []

    def n_ctx(self):
        return self.context

    def tokenize(self, value, **_):
        return list(value)

    def detokenize(self, tokens):
        return bytes(tokens)

    def create_completion(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.options.append(kwargs)
        return {"choices": [{"text": self.text}], "usage": {"completion_tokens": 7}}


def one_chunk_per_document(documents, _chunk_size, _chunk_overlap):
    return [DocumentChunk(document, 0, document.text, 0) for document in documents]


class TtyBuffer(io.StringIO):
    def isatty(self):
        return True


class FakeSearch:
    def __init__(self, values, limits):
        self.values = values
        self.limits = limits
        self.count = len(values)

    def where(self, _):
        return self

    def limit(self, count):
        self.count = count
        self.limits.append(count)
        return self

    def to_list(self):
        return self.values[:self.count]


class FakeTable:
    schema = SimpleNamespace(names=["collection"])

    def __init__(self, fts, vector):
        self.fts = fts
        self.vector = vector
        self.limits = []

    def search(self, _, query_type=None, **__):
        return FakeSearch(self.fts if query_type == "fts" else self.vector, self.limits)


class FakeDb:
    def __init__(self, table):
        self.table = table

    def list_tables(self):
        return SimpleNamespace(tables=["chunks"])

    def open_table(self, _):
        return self.table


class DocumentIndexerTests(unittest.TestCase):
    def test_interactive_console_keeps_logs_and_progress_together(self):
        stream = TtyBuffer()
        handler = InteractiveConsoleHandler(stream)
        handler.setFormatter(ColoredFormatter("%(levelname)s %(message)s"))
        logger = logging.getLogger("interactive-console-test")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)

        handler.update_progress("Embedding", 0, 2)
        logger.info("phase=embed-start")
        handler.update_progress("Embedding", 1, 2)
        handler.update_progress("Embedding", 2, 2)

        output = stream.getvalue()
        self.assertIn("\033[", output)
        self.assertIn("phase=embed-start", output)
        self.assertIn("50% (1/2)", output)
        self.assertEqual(handler._progress, "")

    def test_chunks_and_vectors_are_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "doc.md"
            source.write_text("# Title\n" + "word " * 1200, encoding="utf-8")
            db_path = root / "db"
            result = DocumentIndexer(DocumentIndexerConfig(
                db_path=db_path,
                chunk_size=50,
                batch_size=2,
            )).index(source, FakeEmbeddingModel(), collection="italia")

            self.assertGreater(result["chunks"], 1)
            db = lancedb.connect(str(db_path))
            self.assertEqual(db.open_table("documents").count_rows(), 1)
            self.assertEqual(db.open_table("chunks").count_rows(), result["chunks"])
            self.assertTrue(all(row["active"] for row in db.open_table("chunks").to_arrow().to_pylist()))
            self.assertEqual(
                db.open_table("documents").to_arrow().to_pylist()[0]["collection"], "italia")
            self.assertEqual(
                DocumentIndexer(DocumentIndexerConfig(db_path=db_path)).list_collections(),
                ["italia"])
            matches = DocumentIndexer(DocumentIndexerConfig(db_path=db_path)).search(
                "Title", collections=["italia"], model=FakeEmbeddingModel())
            self.assertTrue(matches)
            self.assertTrue(DocumentIndexer(DocumentIndexerConfig(db_path=db_path)).search(
                "Title", collections=["italia"], mode="fts"))
            self.assertTrue(DocumentIndexer(DocumentIndexerConfig(db_path=db_path)).search(
                "Title", collections=["italia"], mode="hybrid", model=FakeEmbeddingModel(),
                rerank=False))

    def test_unchanged_ingest_skips_chunking_and_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "doc.md"
            source.write_text("unchanged", encoding="utf-8")
            indexer = DocumentIndexer(DocumentIndexerConfig(db_path=root / "db"))
            with patch("src.document_indexer.chunk_documents", one_chunk_per_document):
                indexer.index(source, FakeEmbeddingModel())
            db = lancedb.connect(str(root / "db"))
            chunks_table = db.open_table("chunks")
            inactive = chunks_table.to_arrow().to_pylist()[0]
            chunks_table.add([{**inactive, "id": "legacy-inactive", "active": False}])
            with patch("src.document_indexer.chunk_documents",
                       side_effect=AssertionError("must not chunk")), \
                    patch("src.document_indexer.load_embedding_model",
                          side_effect=AssertionError("must not load model")):
                result = indexer.index(source)

            self.assertEqual(result["unchanged"], 1)
            self.assertEqual(result["embedded"], 0)
            self.assertEqual(lancedb.connect(str(root / "db")).open_table(
                "chunks").count_rows(), 1)

    def test_chunk_size_changes_fingerprint_and_forces_reingest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "doc.md"
            source.write_text("same", encoding="utf-8")
            db_path = root / "db"
            with patch("src.document_indexer.chunk_documents", one_chunk_per_document):
                DocumentIndexer(DocumentIndexerConfig(
                    db_path=db_path, chunk_size=10)).index(source, FakeEmbeddingModel())
                result = DocumentIndexer(DocumentIndexerConfig(
                    db_path=db_path, chunk_size=11)).index(source, FakeEmbeddingModel())
            self.assertEqual(result["unchanged"], 0)
            self.assertEqual(result["embedded"], 1)
            db = lancedb.connect(str(db_path))
            self.assertEqual(db.open_table("documents").count_rows(), 1)
            self.assertEqual(db.open_table("chunks").count_rows(), 1)

    def test_overlap_uses_natural_suffix_with_token_limit(self):
        tokenizer = SimpleNamespace(count_tokens=lambda text: len(text.split()))
        text = "First paragraph has context. Second sentence stays together. Final item."
        self.assertEqual(
            _overlap_suffix(text, tokenizer, 6),
            "Second sentence stays together. Final item.",
        )

        offsets = [(match.start(), match.end())
                   for match in __import__("re").finditer(r"\S+", text)]
        underlying = Mock(return_value={"offset_mapping": offsets})
        fast_tokenizer = SimpleNamespace(get_tokenizer=lambda: underlying)
        self.assertEqual(_overlap_suffix(text, fast_tokenizer, 6),
                         "Second sentence stays together. Final item.")
        underlying.assert_called_once()

    def test_chunk_position_uses_prefix_when_docling_reformats_tail(self):
        source = ("Earlier text.\n8. Where the opinion confirms safeguards in this specific "
                  "regulatory source text. Original continuation.")
        needle = ("8. Where the opinion confirms safeguards in this specific regulatory "
                  "source text. Reformatted continuation.")
        self.assertEqual(
            _chunk_position(source, needle, 0, len(source), 0, "doc.pdf"),
            source.index("8. Where"),
        )

    def test_failed_update_preserves_active_generation(self):
        class FailingEmbeddingModel(FakeEmbeddingModel):
            def create_embedding(self, _value):
                raise RuntimeError("embedding failed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "doc.md"
            source.write_text("old", encoding="utf-8")
            db_path = root / "db"
            indexer = DocumentIndexer(DocumentIndexerConfig(db_path=db_path))
            with patch("src.document_indexer.chunk_documents", one_chunk_per_document):
                indexer.index(source, FakeEmbeddingModel())
                db = lancedb.connect(str(db_path))
                old_id = db.open_table("chunks").to_arrow().to_pylist()[0]["id"]
                old_hash = db.open_table("documents").to_arrow().to_pylist()[0]["content_hash"]
                source.write_text("new", encoding="utf-8")
                result = indexer.index(source, FailingEmbeddingModel())

            active_chunks = db.open_table("chunks").search().where(
                "active = true").limit(10).to_list()
            document = db.open_table("documents").to_arrow().to_pylist()[0]
            self.assertEqual(result["failures"], 1)
            self.assertEqual([row["id"] for row in active_chunks], [old_id])
            self.assertEqual(document["content_hash"], old_hash)
            self.assertEqual(db.open_table("chunks").count_rows(), 1)

    def test_directory_ingest_deactivates_deleted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "a.md").write_text("alpha", encoding="utf-8")
            deleted = corpus / "b.md"
            deleted.write_text("beta", encoding="utf-8")
            db_path = root / "db"
            indexer = DocumentIndexer(DocumentIndexerConfig(db_path=db_path))
            with patch("src.document_indexer.chunk_documents", one_chunk_per_document):
                indexer.index(corpus, FakeEmbeddingModel())
                deleted.unlink()
                result = indexer.index(corpus)

            db = lancedb.connect(str(db_path))
            active_documents = db.open_table("documents").search().where(
                "active = true").limit(10).to_list()
            active_chunks = db.open_table("chunks").search().where(
                "active = true").limit(10).to_list()
            self.assertEqual(result["deleted"], 1)
            self.assertEqual([row["relative_path"] for row in active_documents], ["a.md"])
            self.assertEqual(len(active_chunks), 1)
            self.assertEqual(db.open_table("documents").count_rows(), 1)
            self.assertEqual(db.open_table("chunks").count_rows(), 1)

    def test_collection_registry_updates_only_matching_files_and_can_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "keep.md").write_text("keep", encoding="utf-8")
            (corpus / "skip.txt").write_text("skip", encoding="utf-8")
            indexer = DocumentIndexer(DocumentIndexerConfig(
                db_path=root / "db", index_path=root / "index.yml"))
            entry = indexer.add_collection("docs", corpus)
            self.assertEqual(entry["pattern"], "**/*.md")
            self.assertEqual(json.loads((root / "index.yml").read_text())["collections"]["docs"]["path"],
                             str(corpus.resolve()))
            with patch("src.document_indexer.chunk_documents", one_chunk_per_document):
                result = indexer.update_collections(model=FakeEmbeddingModel())
            self.assertEqual(result[0]["documents"], 1)
            self.assertEqual(len(indexer.list_documents("docs")), 1)
            self.assertEqual(indexer.remove_collection("docs"), 1)
            self.assertEqual(indexer.collections(), [])
            self.assertEqual(indexer.list_documents("docs"), [])
            db = lancedb.connect(str(root / "db"))
            self.assertEqual(db.open_table("documents").count_rows(), 0)
            self.assertEqual(db.open_table("chunks").count_rows(), 0)

    def test_config_file_roundtrip_and_validation(self):
        from src.document_indexer import (
            _apply_config, _config_snapshot, _read_config, _write_config,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            self.assertEqual(_read_config(path), {})
            config = DocumentIndexerConfig()
            updated = _apply_config(config, {
                "model_path": "models/embeddinggemma-300m-Q4_0.gguf",
                "min_score": 0.5, "gpu_layers": 0})
            self.assertEqual(updated.model_path,
                             Path("models/embeddinggemma-300m-Q4_0.gguf"))
            self.assertEqual(updated.min_score, 0.5)
            _write_config(path, _config_snapshot(updated))
            self.assertEqual(_read_config(path)["min_score"], 0.5)
            with self.assertRaisesRegex(ValueError, "Unknown config keys"):
                _apply_config(config, {"nonsense": 1})
            with self.assertRaisesRegex(ValueError, "boolean"):
                _apply_config(config, {"min_score": True})
            with self.assertRaisesRegex(ValueError, "min_score"):
                _apply_config(config, {"min_score": 2.0})
            with self.assertRaises(ValueError):
                _apply_config(config, {"gpu_layers": "turbo"})

    def test_chunk_only_command(self):
        from src.document_indexer import _run_chunk
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "doc.md"
            source.write_text("# Title\n\n" + "some words here. " * 200,
                              encoding="utf-8")
            result = _run_chunk(source, 50, None)
            self.assertGreater(result["count"], 1)
            self.assertEqual(
                set(result["chunks"][0]), {"index", "position", "text"})
            self.assertIn("some words", result["chunks"][0]["text"])
            with self.assertRaisesRegex(ValueError, "chunk_overlap"):
                _run_chunk(source, 50, 50)
            empty = Path(directory) / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "No supported documents"):
                _run_chunk(empty, 50, None)

    def test_pdf_sections_split_on_learn_metadata_and_drop_repeated(self):
        from src.document_indexer import _drop_repeated, _pdf_sections
        text = ("Preamble content.\n"
                "Inheritance in C# Article • 02/03/2023 Body of the first article. "
                "Pattern matching Article • 05/11/2024 Body of the second article.")
        sections = _pdf_sections(text)
        self.assertEqual(len(sections), 3)
        self.assertIn("Preamble", sections[0][2])
        self.assertTrue(sections[1][2].startswith(" Body of the first"))
        self.assertIn("second article", sections[2][2])
        # legal-style fallback still works
        self.assertEqual(len(_pdf_sections("Article 1\nBody\nArticle 2\nMore")), 2)

        def dc(text):
            return DocumentChunk(InputDocument(text, ".", "d.pdf", "h"), 0, text, 0)
        local = [("repeat", 0, dc("x-repeat")), ("repeat", 10, dc("x-repeat")),
                 ("repeat", 20, dc("x-repeat")), ("unique", 0, dc("x-unique")),
                 ("same-section", 0, dc("x-same")), ("same-section", 0, dc("x-same")),
                 ("same-section", 0, dc("x-same"))]
        self.assertEqual([c.text for c in _drop_repeated(local)],
                         ["x-unique", "x-same", "x-same", "x-same"])

    def test_tech_tokens_boilerplate_and_dedup(self):
        from src.document_indexer import _dedup_by_document, _is_boilerplate
        from src.indexer_support import normalize_tech_tokens
        self.assertEqual(normalize_tech_tokens("C# e F# e C++ e .NET"),
                         "csharp e fsharp e cpp e dotnet")
        self.assertIn("csharp", lexicalize_query("Come funziona C#?"))
        self.assertTrue(_is_boilerplate(
            "Indice\n\nIntroduzione .......... 1\nCapitolo 1 ........ 5\n2"))
        self.assertFalse(_is_boilerplate("# Article 33\n\n1. First item"))
        rows = [{"document_id": "a", "id": "1", "chunk_index": 4},
                {"document_id": "a", "id": "2", "chunk_index": 5},
                {"document_id": "a", "id": "2b", "chunk_index": 9},
                {"document_id": "b", "id": "3", "chunk_index": 0}]
        self.assertEqual([row["id"] for row in _dedup_by_document(rows)],
                         ["1", "2b", "3"])

    def test_lexical_query_keeps_all_terms_and_references(self):
        italian = lexicalize_query(
            "Come devono essere costituite le comunità energetiche UE 2019/944?")
        english = lexicalize_query("How must renewable energy communities share electricity?")
        self.assertIn("come", italian)
        self.assertIn("devono", italian)
        self.assertIn("comunita", italian)
        self.assertIn("ue 2019 944", italian)
        self.assertEqual(
            english, "how must renewable energy communities share electricity")
        self.assertEqual(lexicalize_query("How must what"), "how must what")

    def test_icu_fts_finds_italian_and_english_in_one_table(self):
        with tempfile.TemporaryDirectory() as directory:
            db = lancedb.connect(str(Path(directory) / "db"))
            table = db.create_table("chunks", schema=chunks_schema(2))
            base = {
                "document_id": "d", "collection": "mixed", "content_hash": "h",
                "embedding_fingerprint": "e", "chunk_index": 0, "total_chunks": 1,
                "position": 0, "vector": [1.0, 0.0], "active": True,
            }
            table.add([
                {**base, "id": "it", "text": "comunità energetica e cittadini",
                 "text_fts": "comunità energetica e cittadini"},
                {**base, "id": "en", "text": "renewable energy community citizens",
                 "text_fts": "renewable energy community citizens"},
            ])
            _create_fts(table)
            table.add([{**base, "id": "new", "text": "another indexed row",
                        "text_fts": "another indexed row"}])
            self.assertGreater(table.list_indices()[0].num_unindexed_rows, 0)
            _create_fts(table)
            self.assertEqual(table.list_indices()[0].num_unindexed_rows, 0)
            indexer = DocumentIndexer(DocumentIndexerConfig(db_path=Path(directory) / "db"))
            self.assertEqual(indexer.search("comunità cittadini", mode="fts")[0]["id"], "it")
            self.assertEqual(indexer.search("renewable citizens", mode="fts")[0]["id"], "en")

    def test_hybrid_uses_backend_pool_rrf_dedup_and_compact_output(self):
        fts = [{"id": f"f{i}", "document_id": f"d{i}", "collection": "c",
                "text": str(i), "chunk_index": i, "total_chunks": 20, "position": i,
                "_score": float(i)}
               for i in range(20)]
        vector = [{"id": "f19" if i == 19 else f"v{i}", "document_id": f"v{i}",
                   "collection": "c", "text": str(i), "chunk_index": i,
                   "total_chunks": 20, "position": i, "_distance": i / 100}
                  for i in range(20)]
        table = FakeTable(fts, vector)
        with patch("src.document_indexer.lancedb.connect", return_value=FakeDb(table)):
            results = DocumentIndexer().search(
                "query", mode="hybrid", top_k=5, model=FakeEmbeddingModel(), rerank=False)
        self.assertEqual(table.limits, [20, 20])
        self.assertEqual(len({row["id"] for row in results}), len(results))
        self.assertEqual(tuple(results[0]), OUTPUT_FIELDS)
        self.assertNotIn("vector", results[0])
        self.assertIn("v0", {row["id"] for row in results})
        self.assertGreater(results[0]["_hybrid_score"], 2 / 61)

    def test_top_k_above_twenty_expands_both_candidate_pools(self):
        values = [{"id": str(i), "document_id": f"d{i}", "collection": "c",
                   "text": str(i), "chunk_index": i, "total_chunks": 25, "position": i,
                   "_score": 25 - i, "_distance": i / 100} for i in range(25)]
        table = FakeTable(values, values)
        with patch("src.document_indexer.lancedb.connect", return_value=FakeDb(table)):
            results = DocumentIndexer().search(
                "query", mode="hybrid", top_k=25, model=FakeEmbeddingModel(), rerank=False)
        self.assertEqual(table.limits, [25, 25])
        self.assertEqual(len(results), 25)

    def test_reranker_batches_truncates_and_validates_rank_output(self):
        candidates = [{"text": "x" * 1000}, {"text": "short"}]
        reranker = FakeReranker(
            [[0.9, 0.1], [0.2, 0.8]], context=800, batch=400)
        self.assertEqual(_rerank(reranker, "query", candidates), [0.9, 0.2])
        self.assertEqual(len(reranker.prompts), 2)
        self.assertLessEqual(len(reranker.prompts[0].encode()), 400)
        self.assertEqual(reranker.batches, [1, 1])
        with self.assertRaisesRegex(RuntimeError, "incompatible reranker output"):
            _rerank(FakeReranker([[float("nan"), 0.5]]), "q", [{"text": "d"}])

    def test_reranker_scores_duplicate_text_once(self):
        reranker = FakeReranker([[0.8, 0.2], [0.3, 0.7]])
        scores = _rerank(
            reranker, "query", [{"text": "same"}, {"text": "same"}, {"text": "other"}])
        self.assertEqual(scores, [0.8, 0.8, 0.3])
        self.assertEqual(len(reranker.prompts), 2)

    def test_reranker_caps_candidates_and_document_tokens(self):
        values = [
            {"id": str(i), "document_id": f"d{i}", "collection": "c",
             "text": "x" * 100 + str(i), "chunk_index": i, "total_chunks": 5,
             "position": i, "_score": 5 - i, "_distance": i / 10}
            for i in range(5)
        ]
        table = FakeTable(values, values)
        reranker = FakeReranker()
        with patch("src.document_indexer.lancedb.connect", return_value=FakeDb(table)):
            DocumentIndexer(DocumentIndexerConfig(
                rerank_candidates=3, rerank_max_tokens=10)).search(
                    "query", mode="hybrid", top_k=2, model=FakeEmbeddingModel(),
                    rerank=True, reranker=reranker)
        self.assertEqual(len(reranker.prompts), 3)
        self.assertIn("<Document>: " + "x" * 10, reranker.prompts[0])
        self.assertNotIn("<Document>: " + "x" * 11, reranker.prompts[0])

    def test_shell_and_benchmark_reuse_preloaded_models(self):
        indexer = SimpleNamespace()
        indexer.search = Mock(return_value=[{"relative_path": "expected.md"}])
        args = SimpleNamespace(
            no_rerank=False, always_rerank=False, mode="hybrid", collection=None, top_k=5)
        embedding_model = object()
        with patch("src.document_indexer._preload_embedding_model",
                   return_value=embedding_model) as preload, \
                patch("builtins.input", side_effect=["first", "second", "exit"]), \
                patch("src.document_indexer._write_json"):
            _run_shell(indexer, args)
        preload.assert_called_once()
        self.assertEqual(indexer.search.call_count, 2)
        self.assertTrue(all(
            call.kwargs["model"] is embedding_model
            and "reranker" not in call.kwargs
            and call.kwargs["rerank"] is None
            for call in indexer.search.call_args_list))

        with tempfile.TemporaryDirectory() as directory:
            cases = Path(directory) / "cases.json"
            cases.write_text(
                '[{"query":"q","expected_paths":["expected.md"]}]', encoding="utf-8")
            args.cases = cases
            with patch("src.document_indexer._preload_embedding_model",
                       return_value=embedding_model):
                result = _run_benchmark(indexer, args)
        self.assertEqual(result["summary"]["mean_recall"], 1.0)
        self.assertEqual(result["summary"]["mean_reciprocal_rank"], 1.0)

    def test_hybrid_reranks_and_missing_model_suggests_download(self):
        values = [
            {"id": str(i), "document_id": f"d{i}", "collection": "c", "text": str(i),
             "chunk_index": i, "total_chunks": 2, "position": i,
             "_score": 2 - i, "_distance": i / 10}
            for i in range(2)
        ]
        table = FakeTable(values, values)
        with patch("src.document_indexer.lancedb.connect", return_value=FakeDb(table)):
            results = DocumentIndexer().search(
                "query", mode="hybrid", top_k=2, model=FakeEmbeddingModel(),
                rerank=True, reranker=FakeReranker([[0.1, 0.9], [0.9, 0.1]]))
            self.assertEqual(results[0]["id"], "1")
            with self.assertRaisesRegex(RuntimeError, "download_models.ps1 -Model reranker"):
                DocumentIndexer(DocumentIndexerConfig(
                    reranker_model_path=Path("missing.gguf"))).search(
                        "query", mode="hybrid", model=FakeEmbeddingModel(), rerank=True)

    def test_auto_rerank_skips_agreeing_rankings_and_missing_model(self):
        values = [
            {"id": str(i), "document_id": f"d{i}", "collection": "c", "text": str(i),
             "chunk_index": i, "total_chunks": 2, "position": i,
             "_score": 2 - i, "_distance": i / 10}
            for i in range(2)
        ]
        table = FakeTable(values, values)
        indexer = DocumentIndexer(DocumentIndexerConfig(
            reranker_model_path=Path("missing.gguf")))
        with patch("src.document_indexer.lancedb.connect", return_value=FakeDb(table)), \
                patch("src.document_indexer.load_reranker_model",
                      side_effect=AssertionError("must not load")), \
                self.assertLogs("document-indexer", level="INFO") as logs:
            results = indexer.search(
                "query", mode="hybrid", top_k=2, model=FakeEmbeddingModel())
        self.assertTrue(all(result["_rerank_score"] is None for result in results))
        self.assertTrue(any("policy=auto decision=skip reason=top-2-agree" in line
                            for line in logs.output))

        one = FakeTable(values[:1], values[:1])
        with patch("src.document_indexer.lancedb.connect", return_value=FakeDb(one)), \
                patch("src.document_indexer.load_reranker_model",
                      side_effect=AssertionError("must not load")):
            single = indexer.search(
                "query", mode="hybrid", top_k=1, model=FakeEmbeddingModel())
        self.assertIsNone(single[0]["_rerank_score"])

    def test_auto_rerank_runs_on_disagreement_and_reuses_model(self):
        def value(identifier, rank):
            return {"id": identifier, "document_id": "d", "collection": "c",
                    "text": identifier, "chunk_index": rank, "total_chunks": 2,
                    "position": rank, "_score": 2 - rank, "_distance": rank / 10}

        table = FakeTable([value("a", 0), value("b", 1)],
                          [value("b", 0), value("a", 1)])
        reranker = FakeReranker()
        indexer = DocumentIndexer()
        with patch("src.document_indexer.lancedb.connect", return_value=FakeDb(table)), \
                patch("src.document_indexer.load_reranker_model",
                      return_value=reranker) as load:
            first = indexer.search(
                "query", mode="hybrid", top_k=2, model=FakeEmbeddingModel())
            indexer.search("query", mode="hybrid", top_k=2, model=FakeEmbeddingModel())
        load.assert_called_once()
        self.assertTrue(all(result["_rerank_score"] is not None for result in first))

    def test_rerank_overrides_always_and_never(self):
        values = [{"id": "a", "document_id": "d", "collection": "c", "text": "a",
                   "chunk_index": 0, "total_chunks": 1, "position": 0,
                   "_score": 1.0, "_distance": 0.1}]
        table = FakeTable(values, values)
        reranker = FakeReranker()
        with patch("src.document_indexer.lancedb.connect", return_value=FakeDb(table)):
            never = DocumentIndexer().search(
                "query", mode="hybrid", model=FakeEmbeddingModel(),
                rerank=False, reranker=reranker)
            always = DocumentIndexer().search(
                "query", mode="hybrid", model=FakeEmbeddingModel(),
                rerank=True, reranker=reranker)
        self.assertIsNone(never[0]["_rerank_score"])
        self.assertIsNotNone(always[0]["_rerank_score"])

    def test_pdf_articles_are_chunked_separately_with_positions(self):
        text = (
            "Preface\n\nArticle 33\n1. First item\n2. Second item\n\n"
            "Article 34\n1. New first\n2. New second\n\n"
            "Article 35\n1. Last first\n2. Last second\n"
        )
        document = InputDocument(text, ".", "synthetic.pdf", "hash")
        chunks = chunk_documents([document], 50, 8)
        article_chunks = [chunk for chunk in chunks if "Article " in chunk.text]
        self.assertEqual(len(article_chunks), 3)
        self.assertTrue(all(chunk.text.count("Article ") == 1 for chunk in article_chunks))
        self.assertTrue(all("1." in chunk.text for chunk in article_chunks))
        positions = [chunk.position for chunk in chunks]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(any(position > 0 for position in positions))

    def test_office_and_open_document_formats_are_converted_to_markdown(self):
        suffixes = (".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
                    ".odt", ".ods", ".odp")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in suffixes:
                (root / f"sample{suffix}").write_bytes(b"binary")
            documents = read_input_documents(root)

            self.assertEqual({Path(item.relative_path).suffix for item in documents},
                             set(suffixes))
            self.assertTrue(all(not item.text for item in documents))
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter
            converted = DocumentConverter().convert_string(
                "# Report\n\n1. Item", InputFormat.MD, name="sample")
            with patch("docling.document_converter.DocumentConverter") as converter:
                converter.return_value.convert.return_value = converted
                chunks = chunk_documents([documents[1]], 50, 8)
                converter.return_value.convert.assert_called_once()
            self.assertTrue(any("Report" in chunk.text and "1. Item" in chunk.text
                                for chunk in chunks))

    def test_large_sections_use_line_chunker(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "large.md"
            source.write_text("# Title\n\nBody", encoding="utf-8")
            document = InputDocument(source.read_text(), directory, source.name, "hash")
            with patch("src.document_indexer.LARGE_SECTION_CHARS", 1), \
                    self.assertLogs("document-indexer", level="INFO") as logs:
                chunks = chunk_documents([document], 50, 8)
        self.assertTrue(chunks)
        self.assertTrue(any("chunker=line" in line for line in logs.output))

    def test_text_documents_chunk_from_loaded_content(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "document.md"
            source.write_text("# Loaded\n\nContent in memory.", encoding="utf-8")
            document = read_input_documents(source)[0]
            source.unlink()
            chunks = chunk_documents([document], 50, 8)
        self.assertTrue(any("Content in memory" in chunk.text for chunk in chunks))

    def test_vector_and_fts_never_load_reranker(self):
        values = [{"id": "1", "document_id": "d", "collection": "c", "text": "query",
                   "chunk_index": 0, "total_chunks": 1, "position": 0,
                   "_score": 1.0, "_distance": 0.1}]
        table = FakeTable(values, values)
        with patch("src.document_indexer.lancedb.connect", return_value=FakeDb(table)), \
                patch("src.document_indexer.load_reranker_model",
                      side_effect=AssertionError("must not load")):
            DocumentIndexer().search("query", mode="fts")
            DocumentIndexer().search("query", mode="vector", model=FakeEmbeddingModel())

    def test_answer_cites_sources_and_reuses_chat_model(self):
        results = [
            {"title": "Variables", "relative_path": "csharp/variables.md",
             "text": "Declare a variable with a type and a name: int count;"},
            {"title": "Constants", "relative_path": "csharp/constants.md",
             "text": "Use const for compile-time constants."},
        ]
        chat = FakeChatModel()
        indexer = DocumentIndexer()
        with patch("src.document_indexer.load_chat_model", return_value=chat) as load:
            first = indexer.generate_answer("How do I declare a variable?", results)
            indexer.generate_answer("How do I declare a constant?", results)
        load.assert_called_once()
        self.assertEqual(first["answer"], chat.text)
        self.assertEqual(first["sources"], results)
        self.assertIn("[1] Variables", chat.prompts[0])
        self.assertIn("[2] Constants", chat.prompts[0])
        self.assertIn("Question: How do I declare a variable?", chat.prompts[0])
        self.assertEqual(chat.options[0]["max_tokens"], 512)
        self.assertIn("<|im_end|>", chat.options[0]["stop"])

    def test_answer_handles_empty_results_and_missing_model(self):
        empty = DocumentIndexer().generate_answer("q", [], FakeChatModel())
        self.assertEqual(empty["sources"], [])
        self.assertIn("No indexed passage", empty["answer"])
        missing = DocumentIndexer(DocumentIndexerConfig(
            chat_model_path=Path("missing.gguf")))
        with self.assertRaisesRegex(RuntimeError, "download_models.ps1 -Model chat"):
            missing.generate_answer("q", [{"title": "t", "text": "body"}])

    def test_answer_truncates_sources_to_fit_budget(self):
        chat = FakeChatModel(context=1024)
        indexer = DocumentIndexer(DocumentIndexerConfig(
            chat_context=1024, answer_max_tokens=128))
        results = [{"title": "Long", "relative_path": "long.md", "text": "word " * 500}]
        generated = indexer.generate_answer("q", results, chat)
        budget = 1024 - 128 - 32
        self.assertLessEqual(len(chat.prompts[0].encode("utf-8")), budget)
        self.assertNotIn("word " * 500, chat.prompts[0])
        self.assertEqual(generated["sources"], results)

    def test_answer_runs_search_then_generation(self):
        values = [{"id": "a", "document_id": "d", "collection": "c",
                   "text": "declare with type and name", "chunk_index": 0,
                   "total_chunks": 1, "position": 0, "_distance": 0.1}]
        table = FakeTable([], values)
        chat = FakeChatModel()
        indexer = DocumentIndexer()
        with patch("src.document_indexer.lancedb.connect", return_value=FakeDb(table)):
            result = indexer.answer("How do I declare a variable in C#?",
                                    mode="vector", model=FakeEmbeddingModel(),
                                    chat_model=chat)
        self.assertEqual(result["query"], "How do I declare a variable in C#?")
        self.assertEqual(result["answer"], chat.text)
        self.assertEqual(result["sources"][0]["id"], "a")
        self.assertIn("declare with type and name", chat.prompts[0])

    def test_shell_ask_prefix_generates_answer(self):
        indexer = SimpleNamespace()
        indexer.answer = Mock(return_value={"query": "q", "answer": "a", "sources": []})
        args = SimpleNamespace(
            no_rerank=False, always_rerank=False, mode="hybrid", collection=None, top_k=5)
        with patch("src.document_indexer._preload_embedding_model", return_value=None), \
                patch("builtins.input", side_effect=["/ask declare a variable", "exit"]), \
                patch("src.document_indexer._write_json") as write:
            _run_shell(indexer, args)
        indexer.answer.assert_called_once()
        self.assertEqual(indexer.answer.call_args.args[0], "declare a variable")
        write.assert_called_once_with({"query": "q", "answer": "a", "sources": []})


if __name__ == "__main__":
    unittest.main()
