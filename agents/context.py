"""Нормализация и форматирование RAG-контекста.

RAG возвращает ``RetrievedChunk``. Агент переводит их в ``ContextChunk``,
объединяет результаты нескольких подзапросов, убирает дубли и готовит
компактный текст для промптов.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from agents.state import ContextChunk

__all__ = [
    "format_context",
    "format_grouped_context",
    "merge_contexts",
    "metadata_hints",
    "normalize_chunk",
]

_METADATA_KEYS = (
    "doc_number",
    "edition",
    "status",
    "date",
    "document_date",
    "base_doc",
)


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _as_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_chunk(chunk: Any, subquery_id: int) -> ContextChunk:
    """Привести ``RetrievedChunk`` (или совместимый объект) к ``ContextChunk``."""
    if isinstance(chunk, (tuple, list)) and len(chunk) >= 2:
        doc_id = _as_text(chunk[0])
        chunk_id = doc_id
        text = _as_text(chunk[1])
        metadata: dict[str, Any] = {}
        score = 0.0
    else:
        doc_id = _as_text(getattr(chunk, "doc_id", ""))
        chunk_id = _as_text(getattr(chunk, "chunk_id", "")) or doc_id
        metadata = _as_metadata(getattr(chunk, "metadata", None))
        text = _as_text(getattr(chunk, "text", ""))
        score = _as_float(getattr(chunk, "score", 0.0))

        if not text:
            text = _as_text(getattr(chunk, "page_content", ""))

    return ContextChunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        text=text,
        metadata=metadata,
        score=score,
        subquery_ids=[subquery_id],
    )


def merge_contexts(chunks: list[ContextChunk], max_chunks: int) -> list[ContextChunk]:
    """Объединить фрагменты по ``chunk_id`` и ограничить их количество."""
    merged: dict[str, ContextChunk] = {}
    for chunk in chunks:
        key = chunk.chunk_id or f"{chunk.doc_id}:{len(merged)}"
        existing = merged.get(key)
        if existing is None:
            merged[key] = chunk.model_copy(deep=True)
            continue

        for subquery_id in chunk.subquery_ids:
            if subquery_id not in existing.subquery_ids:
                existing.subquery_ids.append(subquery_id)

    result = list(merged.values())
    if max_chunks > 0:
        result = result[:max_chunks]
    return result


def metadata_hints(chunk: ContextChunk) -> dict[str, str]:
    """Извлечь подсказки о редакции и статусе документа.

    Метаданные RAG не гарантируют наличие этих полей, поэтому часть
    сведений аккуратно вытаскивается из ``title`` и ``doc_id``.
    """
    metadata = chunk.metadata
    hints: dict[str, str] = {}

    for key in _METADATA_KEYS:
        value = _as_text(metadata.get(key))
        if value:
            hints[key] = value

    source = " ".join(
        part
        for part in (
            _as_text(metadata.get("title")),
            chunk.doc_id,
            chunk.chunk_id,
            _as_text(metadata.get("source")),
        )
        if part
    )
    lowered = source.lower()

    if "edition" not in hints:
        match = re.search(r"(?:ред(?:акция)?\.?|редакции)\s*(?:от\s*)?(\d{2}\.\d{2}\.\d{4}|\d{4})", source, re.I)
        if match:
            hints["edition"] = match.group(1)

    if "status" not in hints and ("утратил" in lowered or "недейств" in lowered):
        hints["status"] = "утратил силу"

    if "date" not in hints:
        match = re.search(r"\bот\s+(\d{2}\.\d{2}\.\d{4})\b", source)
        if match:
            hints["date"] = match.group(1)

    if "doc_number" not in hints:
        match = re.search(r"\b([А-ЯA-Z]{1,4}[-/]\d{2,4}[-/]\d{2,4})\b", source)
        if match:
            hints["doc_number"] = match.group(1)

    return hints


def _format_chunk(chunk: ContextChunk) -> str:
    lines: list[str] = []
    title = _as_text(chunk.metadata.get("title"))
    if title:
        lines.append(f"title: {title}")
    if chunk.doc_id:
        lines.append(f"doc_id: {chunk.doc_id}")
    if chunk.chunk_id:
        lines.append(f"chunk_id: {chunk.chunk_id}")
    if chunk.score:
        lines.append(f"score: {chunk.score:.4f}")

    for key, value in metadata_hints(chunk).items():
        lines.append(f"{key}: {value}")

    lines.append(chunk.text.strip())
    return "\n".join(line for line in lines if line)


def format_context(chunks: list[ContextChunk], max_chars: int = 0) -> str:
    """Собрать фрагменты в текст, соблюдая ограничение длины."""
    blocks: list[str] = []
    current_length = 0

    for chunk in chunks:
        block = _format_chunk(chunk)
        if not block:
            continue

        separator = 2 if blocks else 0
        if max_chars > 0 and current_length + separator + len(block) > max_chars:
            remaining = max_chars - current_length - separator
            if remaining > 0:
                blocks.append(block[:remaining].rstrip())
            break

        blocks.append(block)
        current_length += separator + len(block)

    return "\n\n".join(blocks)


def format_grouped_context(
    subqueries: list[Any],
    chunks: list[ContextChunk],
    max_chars: int = 0,
) -> str:
    """Сгруппировать контекст по подзапросам для промпта-оценщика."""
    blocks: list[str] = []
    current_length = 0
    assigned: set[int] = set()

    for subquery in subqueries:
        selected = [
            chunk for chunk in chunks if subquery.id in chunk.subquery_ids
        ]
        assigned.update(id(chunk) for chunk in selected)
        body = format_context(selected) or "Фрагменты не найдены."
        block = f"### Подзапрос {subquery.id}: {subquery.text}\n{body}"

        separator = 2 if blocks else 0
        if max_chars > 0 and current_length + separator + len(block) > max_chars:
            remaining = max_chars - current_length - separator
            if remaining > 0:
                blocks.append(block[:remaining].rstrip())
            break

        blocks.append(block)
        current_length += separator + len(block)

    extra = [chunk for chunk in chunks if id(chunk) not in assigned]
    if extra:
        block = "### Прочие фрагменты\n" + format_context(extra)
        separator = 2 if blocks else 0
        if max_chars <= 0 or current_length + separator + len(block) <= max_chars:
            blocks.append(block)

    return "\n\n".join(blocks)
