from __future__ import annotations

import unittest

from agents.rag import RagClient
from rag.index.contracts import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return self.chunks[:k]


def make_chunk(**overrides) -> RetrievedChunk:
    data = {
        "doc_id": "doc-1",
        "chunk_id": "chunk-1",
        "text": "Текст первого чанка",
        "metadata": {
            "title": "Заголовок",
            "doc_type": "Федеральный закон",
            "docdate": "01.01.2024",
        },
        "score": 0.9,
    }
    data.update(overrides)
    return RetrievedChunk(**data)


class RagClientTest(unittest.TestCase):
    def test_search_uses_hybrid_and_passes_query(self) -> None:
        retriever = FakeRetriever([make_chunk()])
        client = RagClient({"hybrid": retriever})

        result = client.search("налоги", 3)

        self.assertEqual(result, [make_chunk()])
        self.assertEqual(retriever.calls, [("налоги", 3)])

    def test_search_prefers_hybrid(self) -> None:
        hybrid = FakeRetriever([make_chunk(chunk_id="hybrid")])
        dense = FakeRetriever([make_chunk(chunk_id="dense")])
        client = RagClient({"hybrid": hybrid, "dense": dense})

        self.assertEqual(client.search("q", 1)[0].chunk_id, "hybrid")

    def test_search_falls_back_to_dense_then_bm25(self) -> None:
        dense = RagClient({"dense": FakeRetriever([make_chunk(chunk_id="dense")])})
        self.assertEqual(dense.search("q", 1)[0].chunk_id, "dense")

        bm25 = RagClient({"bm25": FakeRetriever([make_chunk(chunk_id="bm25")])})
        self.assertEqual(bm25.search("q", 1)[0].chunk_id, "bm25")

    def test_empty_query_returns_empty_without_retriever(self) -> None:
        self.assertEqual(RagClient({}).search("   ", 5), [])

    def test_non_positive_k_returns_empty(self) -> None:
        self.assertEqual(RagClient({"hybrid": FakeRetriever([make_chunk()])}).search("q", 0), [])

    def test_missing_retriever_raises(self) -> None:
        with self.assertRaises(ValueError):
            RagClient({"unknown": FakeRetriever([])}).search("q", 1)

    def test_format_context_contains_source_fields(self) -> None:
        context = RagClient({"hybrid": FakeRetriever([])}).format_context([make_chunk()])

        self.assertIn("[Источник 1]", context)
        self.assertIn("doc_id: doc-1", context)
        self.assertIn("chunk_id: chunk-1", context)
        self.assertIn("heading: Заголовок", context)
        self.assertIn("doc_type: Федеральный закон", context)
        self.assertIn("doc_date: 01.01.2024", context)
        self.assertIn("text: Текст первого чанка", context)

    def test_to_sources_deduplicates_and_uses_fallbacks(self) -> None:
        chunks = [
            make_chunk(),
            make_chunk(score=0.5),
            make_chunk(
                chunk_id="chunk-2",
                metadata={
                    "heading": "Fallback heading",
                    "doc_typeIPS": "Кодекс",
                    "docdateIPS": "02.02.2024",
                },
                score=0.7,
            ),
        ]

        sources = RagClient({"hybrid": FakeRetriever([])}).to_sources(chunks)

        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["heading"], "Заголовок")
        self.assertEqual(sources[1]["heading"], "Fallback heading")
        self.assertEqual(sources[1]["doc_type"], "Кодекс")
        self.assertEqual(sources[1]["doc_date"], "02.02.2024")


if __name__ == "__main__":
    unittest.main()
