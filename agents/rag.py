"""Адаптер между агентом и RAG-индексом.

Агент не знает, как устроены BM25, FAISS и Hybrid. Он получает уже готовые
retriever'ы от бэкенда и работает с ними через узкий интерфейс.
"""

from __future__ import annotations

from typing import Any, Protocol

from agents.state import Source
from rag.index.contracts import RetrievedChunk

__all__ = ["RagClient", "RetrieverLike"]

_RETRIEVER_PREFERENCE = ("hybrid", "dense", "bm25")


class RetrieverLike(Protocol):
    """Минимальный контракт retriever'а, который нужен агенту."""

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        ...


class RagClient:
    """Тонкая обёртка над готовыми retriever'ами.

    Parameters
    ----------
    retrievers:
        Словарь retriever'ов, построенных бэкендом при старте приложения.
        Ключ ``"hybrid"`` — основной, остальные доступны как fallback.
    """

    def __init__(self, retrievers: dict[str, RetrieverLike]) -> None:
        self._retrievers = retrievers

    def _select_retriever(self) -> RetrieverLike:
        """Выбрать лучший доступный retriever."""
        for name in _RETRIEVER_PREFERENCE:
            retriever = self._retrievers.get(name)
            if retriever is not None:
                return retriever

        available = ", ".join(sorted(self._retrievers)) or "нет"
        expected = ", ".join(_RETRIEVER_PREFERENCE)
        raise ValueError(
            f"Не найден подходящий retriever. Доступные: {available}. "
            f"Ожидается один из: {expected}."
        )

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Найти top-k чанков по запросу."""
        if not query or not query.strip() or k <= 0:
            return []
        return self._select_retriever().retrieve(query.strip(), k)

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Собрать контекст для промпта LLM."""
        blocks: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.metadata or {}
            blocks.append(
                "\n".join(
                    [
                        f"[Источник {index}]",
                        f"doc_id: {chunk.doc_id}",
                        f"chunk_id: {chunk.chunk_id}",
                        f"heading: {_first(metadata, 'title', 'heading', 'headingIPS')}",
                        f"doc_type: {_first(metadata, 'doc_type', 'doc_typeIPS')}",
                        f"doc_date: {_first(metadata, 'docdate', 'docdateIPS')}",
                        f"text: {chunk.text}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def to_sources(self, chunks: list[RetrievedChunk]) -> list[Source]:
        """Превратить чанки в компактный список источников для SSE."""
        sources: list[Source] = []
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            metadata = chunk.metadata or {}
            sources.append(
                Source(
                    doc_id=str(chunk.doc_id),
                    chunk_id=str(chunk.chunk_id),
                    heading=_first(metadata, "title", "heading", "headingIPS"),
                    doc_type=_first(metadata, "doc_type", "doc_typeIPS"),
                    doc_date=_first(metadata, "docdate", "docdateIPS"),
                    score=float(chunk.score),
                )
            )
        return sources


def _first(metadata: dict[str, Any], *keys: str) -> str:
    """Вернуть первое непустое значение из метаданных."""
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""
