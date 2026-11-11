from __future__ import annotations

import unittest

from agents.rag import RagClient


class FakeSearch:
    def __init__(self, context: list[tuple[str, str]]) -> None:
        self.context = context
        self.calls: list[tuple[str, object]] = []

    def __call__(self, query: str, retrievers: object) -> list[tuple[str, str]]:
        self.calls.append((query, retrievers))
        return self.context


class RagClientTest(unittest.TestCase):
    def test_search_passes_stripped_query_and_retrievers(self) -> None:
        retrievers = object()
        context = [("doc-1", "готовый контекст")]
        search = FakeSearch(context)
        client = RagClient(retrievers, search_fn=search)

        result = client.search("  налоги  ")

        self.assertEqual(result, context)
        self.assertEqual(search.calls, [("налоги", retrievers)])

    def test_empty_query_returns_empty_without_search(self) -> None:
        search = FakeSearch([("doc-1", "контекст")])
        client = RagClient(object(), search_fn=search)

        self.assertEqual(client.search("   "), [])
        self.assertEqual(search.calls, [])


if __name__ == "__main__":
    unittest.main()
