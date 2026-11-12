from __future__ import annotations

import unittest

from agents.context import (
    format_context,
    merge_contexts,
    metadata_hints,
    normalize_chunk,
)
from agents.state import ContextChunk
from rag.index.contracts import RetrievedChunk


class NormalizeChunkTest(unittest.TestCase):
    def test_converts_retrieved_chunk(self) -> None:
        chunk = normalize_chunk(
            RetrievedChunk(
                doc_id="doc-1",
                chunk_id="chunk-1",
                text="Текст",
                metadata={"title": "Договор"},
                score=0.5,
            ),
            2,
        )

        self.assertEqual(chunk.doc_id, "doc-1")
        self.assertEqual(chunk.chunk_id, "chunk-1")
        self.assertEqual(chunk.metadata, {"title": "Договор"})
        self.assertEqual(chunk.subquery_ids, [2])

    def test_chunk_id_falls_back_to_doc_id(self) -> None:
        chunk = normalize_chunk(
            RetrievedChunk(
                doc_id="doc-1",
                chunk_id="",
                text="Текст",
                metadata={},
                score=0.1,
            ),
            1,
        )

        self.assertEqual(chunk.chunk_id, "doc-1")


class MergeContextsTest(unittest.TestCase):
    def test_deduplicates_by_chunk_id_and_merges_subqueries(self) -> None:
        first = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Текст",
            subquery_ids=[1],
        )
        second = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Текст",
            subquery_ids=[2],
        )

        merged = merge_contexts([first, second], max_chunks=10)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].subquery_ids, [1, 2])

    def test_respects_max_chunks(self) -> None:
        chunks = [
            ContextChunk(doc_id=f"doc-{index}", chunk_id=f"chunk-{index}", text="Т")
            for index in range(5)
        ]

        merged = merge_contexts(chunks, max_chunks=2)

        self.assertEqual([chunk.doc_id for chunk in merged], ["doc-0", "doc-1"])


class FormatContextTest(unittest.TestCase):
    def test_includes_sources_and_metadata(self) -> None:
        chunk = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Текст нормы",
            metadata={"title": "Налоговый кодекс", "edition": "2024"},
            score=0.9,
            subquery_ids=[1],
        )

        formatted = format_context([chunk])

        self.assertIn("title: Налоговый кодекс", formatted)
        self.assertIn("doc_id: doc-1", formatted)
        self.assertIn("edition: 2024", formatted)
        self.assertIn("Текст нормы", formatted)

    def test_truncates_by_max_chars(self) -> None:
        chunk = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Т" * 200,
            metadata={},
            subquery_ids=[1],
        )

        formatted = format_context([chunk], max_chars=80)

        self.assertLessEqual(len(formatted), 80)

    def test_metadata_hints_extract_status_and_edition(self) -> None:
        chunk = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Текст",
            metadata={"title": "Договор утратил силу, редакция 2024"},
            subquery_ids=[1],
        )

        hints = metadata_hints(chunk)

        self.assertEqual(hints["status"], "утратил силу")
        self.assertEqual(hints["edition"], "2024")


if __name__ == "__main__":
    unittest.main()
