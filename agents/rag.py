"""Адаптер между агентом и RAG-индексом.

Единственная точка входа в RAG — ``rag.rag_pipeline.search_chunks``.
Retriever'ы приходят в агент снаружи, поэтому агент не знает, как устроены
BM25, FAISS и Hybrid.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["RagClient"]

SearchFn = Callable[[str, Any], str]


class RagClient:
    """Тонкая обёртка над ``search_chunks``.

    Parameters
    ----------
    retrievers:
        Готовые retriever'ы RAG, построенные снаружи до вызова агента.
    search_fn:
        Функция поиска для подмены в тестах. По умолчанию используется
        ``rag.rag_pipeline.search_chunks``.
    """

    def __init__(
        self,
        retrievers: Any,
        search_fn: SearchFn | None = None,
    ) -> None:
        self._retrievers = retrievers
        self._search_fn = search_fn

    def search(self, query: str) -> str:
        """Получить готовый контекст по запросу."""
        if not query or not query.strip():
            return ""

        search_fn = self._search_fn
        if search_fn is None:
            from rag.rag_pipeline import search_chunks

            search_fn = search_chunks

        return search_fn(query.strip(), self._retrievers)
