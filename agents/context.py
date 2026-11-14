"""Нормализация и форматирование RAG-контекста.

RAG возвращает ``RetrievedChunk``. Агент переводит их в ``ContextChunk``,
обогащает метаданные документа, объединяет результаты нескольких подзапросов,
убирает дубли и готовит компактный текст для промптов.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from agents.state import ContextChunk

__all__ = [
    "document_title",
    "format_context",
    "format_grouped_context",
    "merge_contexts",
    "metadata_hints",
    "normalize_chunk",
    "source_label",
]

_METADATA_HINT_KEYS = (
    "doc_number",
    "document_date",
    "edition",
    "status",
    "base_doc",
)

_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}
_MONTH_NAMES = "|".join(_MONTHS)

_CONTRACT_NUMBER_RES = (
    re.compile(
        r"[А-ЯЁA-Z]{1,4}[-/]\d{2,4}(?:[-/]\d{2,4}|[-/][А-ЯЁA-Z]{1,4})?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\d{2,4}[-/]\d{2,4}[-/][А-ЯЁA-Z]{1,4}",
        re.IGNORECASE,
    ),
    re.compile(r"\d{2,4}[-/]\d{2,4}", re.IGNORECASE),
    re.compile(r"\d{1,4}-[А-ЯЁA-Z]{2,4}", re.IGNORECASE),
)
_GENERIC_NUMBER_RE = re.compile(
    r"(?:№|N)\s*([А-ЯЁA-Z0-9][А-ЯЁA-Z0-9./-]*)",
    re.IGNORECASE,
)
_DASH_DATE_RE = re.compile(
    rf"[—–]\s*(\d{{1,2}}\s+(?:{_MONTH_NAMES})\s+\d{{4}}|\d{{1,2}}\.\d{{2}}\.\d{{4}})"
    r"(?:\s*г\.)?",
    re.IGNORECASE,
)
_FROM_DATE_RE = re.compile(r"\bот\s+(\d{1,2}\.\d{2}\.\d{4})\b", re.IGNORECASE)
_MONTH_DATE_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({_MONTH_NAMES})\s+(\d{{4}})\b",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2}\.\d{2}\.\d{4})\b")
_EDITION_RE = re.compile(
    r"ред(?:акция)?\.?\s*(?:от\s*)?(\d{1,2}\.\d{2}\.\d{4}|\d{4})",
    re.IGNORECASE,
)
_INVALID_STATUS_PATTERNS = (
    re.compile(r"утратил(?:а)?\s+силу", re.IGNORECASE),
    re.compile(r"утрачивает\s+силу", re.IGNORECASE),
    re.compile(r"не\s+действу(?:ет|ющ)", re.IGNORECASE),
    re.compile(r"недейств", re.IGNORECASE),
)
_ACTIVE_STATUS_RE = re.compile(r"\bдействует\b", re.IGNORECASE)

_LAW_PREFIX = "законы__"
_COMPANY_PREFIX = "документы компании__"
_LAW_LABEL_RE = re.compile(
    r"^(?P<code>.+?)_ст(?P<article>\d+(?:\.\d+)*)(?:_|$)",
    re.IGNORECASE,
)
_FZ_CODE_RE = re.compile(r"^(\d+)ФЗ$", re.IGNORECASE)
_CONTRACT_HEAD_RE = re.compile(
    r"^(?P<kind>ДОГОВОР[^№]*?)\s*№\s*(?P<number>[^\s,]+)",
    re.IGNORECASE,
)
_ADDENDUM_HEAD_RE = re.compile(
    r"^(?P<kind>ДОПОЛНИТЕЛЬНОЕ\s+СОГЛАШЕНИЕ)\s*№\s*(?P<number>[^\s,]+)"
    r"(?:\s+к\s+договору(?:\s+[А-ЯЁA-Z]+)?\s*№\s*(?P<base>[^\s,]+))?",
    re.IGNORECASE,
)
_ADDENDUM_STEM_RE = re.compile(
    r"^(?:Доп\.?|Дополнительное)\s+соглашение\s+(\d+)\s+к\s+договору\s+(.+)$",
    re.IGNORECASE,
)
_REGULATION_STEM_RE = re.compile(
    r"^(?P<name>Регламент[^()]*?)\s+ред\.?\s*(?P<edition>\d{4})\s*"
    r"(?P<status>\([^)]*\))?$",
    re.IGNORECASE,
)
_COMPACT_CONTRACT_NUMBER_RE = re.compile(
    r"^([А-ЯЁA-Z]{1,8}-\d{1,4})-(\d{2,4})$",
    re.IGNORECASE,
)


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _as_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _as_optional_int(value: Any, *, min_value: int | None = None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    if min_value is not None and result < min_value:
        return None
    return result


def _format_date(day: int, month: int, year: int) -> str | None:
    if not 1 <= day <= 31 or not 1 <= month <= 12:
        return None
    return f"{day:02d}.{month:02d}.{year:04d}"


def _parse_numeric_date(value: str) -> str | None:
    match = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", value)
    if match is None:
        return None
    return _format_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _parse_month_date(value: str) -> str | None:
    match = _MONTH_DATE_RE.search(value)
    if match is None:
        return None
    month = _MONTHS.get(match.group(2).lower())
    if month is None:
        return None
    return _format_date(int(match.group(1)), month, int(match.group(3)))


def _parse_date_text(value: str) -> str | None:
    return _parse_numeric_date(value) or _parse_month_date(value)


def _normalize_date_value(value: Any) -> str | None:
    return _parse_date_text(_as_text(value))


def _normalize_edition_value(value: Any) -> str | None:
    text = _as_text(value)
    if not text:
        return None
    date_value = _parse_date_text(text)
    if date_value:
        return date_value
    match = re.search(r"\b(\d{4})\b", text)
    if match is not None:
        return match.group(1)
    return text


def _clean_number(value: str) -> str:
    return value.strip().strip(".,;:()[]«»\"'")


def _parse_doc_number(header: str, doc_id: str) -> str | None:
    for pattern in _CONTRACT_NUMBER_RES:
        for source in (header, doc_id):
            match = pattern.search(source or "")
            if match is not None:
                return _clean_number(match.group(0))

    for source in (header, doc_id):
        match = _GENERIC_NUMBER_RE.search(source or "")
        if match is not None:
            return _clean_number(match.group(1))
    return None


def _parse_document_date(header: str) -> str | None:
    if not header:
        return None

    match = _DASH_DATE_RE.search(header)
    if match is not None:
        date_value = _parse_date_text(match.group(1))
        if date_value:
            return date_value

    match = _FROM_DATE_RE.search(header)
    if match is not None:
        date_value = _parse_numeric_date(match.group(1))
        if date_value:
            return date_value

    month_date = _parse_month_date(header)
    if month_date:
        return month_date

    match = _NUMERIC_DATE_RE.search(header)
    if match is not None:
        return _parse_numeric_date(match.group(1))
    return None


def _parse_edition(header: str, doc_id: str) -> str | None:
    for source in (header, doc_id):
        match = _EDITION_RE.search(source or "")
        if match is None:
            continue
        raw = match.group(1)
        if "." in raw:
            date_value = _parse_numeric_date(raw)
            if date_value:
                return date_value
        return raw
    return None


def _parse_status(header: str) -> str | None:
    if not header:
        return None

    matches: list[tuple[int, str]] = []
    for pattern in _INVALID_STATUS_PATTERNS:
        for match in pattern.finditer(header):
            matches.append((match.start(), "утратил силу"))
    for match in _ACTIVE_STATUS_RE.finditer(header):
        matches.append((match.start(), "действует"))

    if not matches:
        return None
    return min(matches, key=lambda item: item[0])[1]


def _normalize_status_value(value: Any) -> str | None:
    text = _as_text(value).lower()
    if not text:
        return None
    if "утратил" in text or "недейств" in text or "не действу" in text:
        return "утратил силу"
    if "действует" in text or "действующ" in text:
        return "действует"
    return None


def _doc_stem(doc_id: str) -> str:
    stem = _as_text(doc_id)
    for prefix in (_LAW_PREFIX, _COMPANY_PREFIX):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    lowered = stem.lower()
    for suffix in (".txt", ".md", ".docx"):
        if lowered.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.strip()


def _normalize_compact_contract_number(value: str) -> str:
    text = _as_text(value)
    match = _COMPACT_CONTRACT_NUMBER_RE.match(text)
    if match is not None:
        return f"{match.group(1)}/{match.group(2)}"
    return text


def _contract_kind(value: str) -> str:
    words = _as_text(value).split()
    if not words:
        return "Договор"
    kind = " ".join(words[:2])
    return kind.capitalize() if kind.isupper() else kind


def _company_stem_label(stem: str) -> str:
    text = _as_text(stem)
    addendum = _ADDENDUM_STEM_RE.match(text)
    if addendum is not None:
        number = _normalize_compact_contract_number(addendum.group(2))
        return f"Доп. соглашение № {addendum.group(1)} к договору № {number}"

    regulation = _REGULATION_STEM_RE.match(text)
    if regulation is not None:
        name = regulation.group("name").strip()
        edition = regulation.group("edition")
        status = (regulation.group("status") or "").lower()
        suffix = ", утратил силу" if "утратил" in status else ""
        return f"{name} (ред. {edition}{suffix})"

    return text


def _company_label(stem: str, header: str) -> str:
    head = _as_text(header)

    addendum = _ADDENDUM_HEAD_RE.match(head)
    if addendum is not None:
        number = _normalize_compact_contract_number(addendum.group("number"))
        base = _normalize_compact_contract_number(addendum.group("base") or "")
        if base:
            return f"Доп. соглашение № {number} к договору № {base}"
        return f"Доп. соглашение № {number}"

    contract = _CONTRACT_HEAD_RE.match(head)
    if contract is not None:
        kind = _contract_kind(contract.group("kind"))
        number = _normalize_compact_contract_number(contract.group("number"))
        return f"{kind} № {number}"

    return _company_stem_label(stem)


def _law_label(stem: str) -> str | None:
    match = _LAW_LABEL_RE.match(stem)
    if match is None:
        return None
    code = _FZ_CODE_RE.sub(r"\1-ФЗ", match.group("code").strip())
    article = match.group("article").strip()
    return f"{code}, ст. {article}"


def _build_source_label(
    doc_id: str,
    header: str,
    metadata: Mapping[str, Any],
) -> str:
    stem = _doc_stem(doc_id)

    if doc_id.startswith(_LAW_PREFIX):
        law = _law_label(stem)
        if law:
            return law

    if doc_id.startswith(_COMPANY_PREFIX):
        return _company_label(stem, header)

    title = _as_text(metadata.get("document_title")) or _as_text(header)
    return title or stem or _as_text(doc_id)


def source_label(chunk: ContextChunk) -> str:
    """Вернуть человекочитаемое название источника."""
    metadata = chunk.metadata or {}
    stored = _as_text(metadata.get("source_label"))
    if stored:
        return stored
    header = (
        _as_text(metadata.get("head_doc"))
        or _as_text(metadata.get("title"))
        or _as_text(metadata.get("document_title"))
    )
    return _build_source_label(chunk.doc_id, header, metadata)


def document_title(chunk: ContextChunk) -> str:
    """Вернуть человекочитаемый заголовок документа."""
    metadata = chunk.metadata
    return (
        _as_text(metadata.get("document_title"))
        or _as_text(metadata.get("head_doc"))
        or _as_text(metadata.get("title"))
        or _as_text(chunk.doc_id)
    )


def enrich_metadata(metadata: Any, doc_id: str) -> dict[str, Any]:
    """Дополнить сырые метаданные чанка полями документа."""
    enriched = _as_metadata(metadata)
    header = _as_text(enriched.get("head_doc")) or _as_text(enriched.get("title"))
    document = header or _as_text(doc_id)

    enriched["document_title"] = document
    enriched["source_label"] = (
        _as_text(enriched.get("source_label"))
        or _build_source_label(doc_id, header, enriched)
    )
    enriched["doc_number"] = (
        _as_text(enriched.get("doc_number"))
        or _parse_doc_number(header, doc_id)
        or None
    )
    enriched["document_date"] = (
        _normalize_date_value(
            enriched.get("document_date") or enriched.get("date")
        )
        or _parse_document_date(header)
        or None
    )
    enriched["edition"] = (
        _normalize_edition_value(enriched.get("edition"))
        or _parse_edition(header, doc_id)
        or None
    )
    enriched["status"] = (
        _normalize_status_value(enriched.get("status"))
        or _parse_status(header)
        or None
    )

    chunk_index = _as_optional_int(enriched.get("chunk_index"), min_value=0)
    char_start = _as_optional_int(enriched.get("char_start"), min_value=0)
    char_end = _as_optional_int(enriched.get("char_end"), min_value=0)
    if char_start is None or char_end is None or char_end <= char_start:
        char_end = None
    enriched["chunk_index"] = chunk_index
    enriched["char_start"] = char_start
    enriched["char_end"] = char_end
    return enriched


def normalize_chunk(chunk: Any, subquery_id: int) -> ContextChunk | None:
    """Привести ``RetrievedChunk`` (или совместимый объект) к ``ContextChunk``.

    Пустые чанки отбрасываются: если после ``strip()`` в тексте ничего не
    осталось, возвращается ``None``.
    """
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

    if not text:
        return None

    return ContextChunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        text=text,
        metadata=enrich_metadata(metadata, doc_id),
        score=score,
        subquery_ids=[subquery_id],
    )


def merge_contexts(chunks: list[ContextChunk], max_chunks: int) -> list[ContextChunk]:
    """Объединить фрагменты по ``chunk_id`` и ограничить их количество."""
    merged: dict[str, ContextChunk] = {}
    for chunk in chunks:
        if not chunk.text.strip():
            continue

        key = chunk.chunk_id or f"{chunk.doc_id}:{len(merged)}"
        existing = merged.get(key)
        if existing is None:
            merged[key] = chunk.model_copy(deep=True)
            continue

        for subquery_id in chunk.subquery_ids:
            if subquery_id not in existing.subquery_ids:
                existing.subquery_ids.append(subquery_id)

        existing.score = max(existing.score, chunk.score)
        for meta_key, meta_value in chunk.metadata.items():
            if meta_value not in (None, "", [], {}) and not existing.metadata.get(
                meta_key
            ):
                existing.metadata[meta_key] = meta_value

    result = list(merged.values())
    if max_chunks > 0:
        result = result[:max_chunks]
    return result


def metadata_hints(chunk: ContextChunk) -> dict[str, str]:
    """Извлечь подсказки о номере, дате, редакции и статусе документа."""
    enriched = enrich_metadata(chunk.metadata, chunk.doc_id)
    hints: dict[str, str] = {}
    for key in _METADATA_HINT_KEYS:
        value = _as_text(enriched.get(key))
        if value:
            hints[key] = value
    return hints


def _chunk_sort_key(chunk: ContextChunk) -> tuple[int, int]:
    index = _as_optional_int(chunk.metadata.get("chunk_index"), min_value=0)
    start = _as_optional_int(chunk.metadata.get("char_start"), min_value=0)
    return (
        index if index is not None else 10**9,
        start if start is not None else 10**9,
    )


def _document_groups(
    chunks: list[ContextChunk],
) -> list[tuple[str, list[ContextChunk]]]:
    groups: dict[str, list[ContextChunk]] = {}
    order: list[str] = []
    for chunk in chunks:
        if not chunk.text.strip():
            continue
        key = chunk.doc_id or chunk.chunk_id
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(chunk)
    return [(key, groups[key]) for key in order]


def _text_overlap(first: str, second: str) -> int:
    limit = min(len(first), len(second), 1000)
    for size in range(limit, 0, -1):
        if first[-size:] == second[:size]:
            return size
    return 0


def _can_merge(current: ContextChunk, following: ContextChunk) -> bool:
    current_start = _as_optional_int(
        current.metadata.get("char_start"), min_value=0
    )
    current_end = _as_optional_int(current.metadata.get("char_end"), min_value=0)
    next_start = _as_optional_int(following.metadata.get("char_start"), min_value=0)
    next_end = _as_optional_int(following.metadata.get("char_end"), min_value=0)
    current_index = _as_optional_int(
        current.metadata.get("chunk_index"), min_value=0
    )
    next_index = _as_optional_int(
        following.metadata.get("chunk_index"), min_value=0
    )

    consecutive = (
        current_index is not None
        and next_index is not None
        and next_index == current_index + 1
    )
    if (
        current_index is not None
        and next_index is not None
        and not consecutive
    ):
        return False

    if None not in (current_start, current_end, next_start, next_end):
        # char_start в RAG считается от начала блока, а не документа,
        # поэтому сброс смещения вперёд означает границу блока.
        if next_start < current_start:
            return False
        return next_start <= current_end + 1

    if consecutive:
        return _text_overlap(current.text, following.text) > 0
    return False


def _merge_two_chunks(current: ContextChunk, following: ContextChunk) -> ContextChunk:
    overlap = _text_overlap(current.text, following.text)
    if overlap:
        text = current.text + following.text[overlap:]
    else:
        text = current.text.rstrip() + "\n" + following.text.lstrip()

    metadata = dict(current.metadata)
    for key in (
        "document_title",
        "head_doc",
        "doc_number",
        "document_date",
        "edition",
        "status",
    ):
        if not metadata.get(key) and following.metadata.get(key):
            metadata[key] = following.metadata[key]

    starts = [
        value
        for value in (
            current.metadata.get("char_start"),
            following.metadata.get("char_start"),
        )
        if isinstance(value, int)
    ]
    ends = [
        value
        for value in (
            current.metadata.get("char_end"),
            following.metadata.get("char_end"),
        )
        if isinstance(value, int)
    ]
    indexes = [
        value
        for value in (
            current.metadata.get("chunk_index"),
            following.metadata.get("chunk_index"),
        )
        if isinstance(value, int)
    ]
    metadata["char_start"] = min(starts) if starts else None
    metadata["char_end"] = max(ends) if ends else None
    metadata["chunk_index"] = min(indexes) if indexes else None

    chunk_ids: list[str] = []
    for chunk in (current, following):
        stored = chunk.metadata.get("chunk_ids")
        if isinstance(stored, (list, tuple)):
            chunk_ids.extend(str(value) for value in stored if value)
        elif chunk.chunk_id:
            chunk_ids.append(chunk.chunk_id)
    metadata["chunk_ids"] = list(dict.fromkeys(chunk_ids))

    subquery_ids = list(current.subquery_ids)
    for subquery_id in following.subquery_ids:
        if subquery_id not in subquery_ids:
            subquery_ids.append(subquery_id)

    return ContextChunk(
        doc_id=current.doc_id,
        chunk_id=current.chunk_id,
        text=text,
        metadata=metadata,
        score=max(current.score, following.score),
        subquery_ids=subquery_ids,
    )


def _merge_document_group(chunks: list[ContextChunk]) -> list[ContextChunk]:
    ordered = sorted(chunks, key=_chunk_sort_key)
    merged: list[ContextChunk] = []
    for chunk in ordered:
        if merged and _can_merge(merged[-1], chunk):
            merged[-1] = _merge_two_chunks(merged[-1], chunk)
        else:
            merged.append(chunk.model_copy(deep=True))
    return merged


def _format_document(doc_id: str, chunks: list[ContextChunk]) -> str:
    if not chunks:
        return ""

    first = chunks[0]
    lines: list[str] = []
    title = document_title(first)
    if title:
        lines.append(f"title: {title}")
    if doc_id:
        lines.append(f"doc_id: {doc_id}")
    label = source_label(first)
    if label:
        lines.append(f"source_label: {label}")

    hints = metadata_hints(first)
    for key in _METADATA_HINT_KEYS:
        value = hints.get(key)
        if value:
            lines.append(f"{key}: {value}")

    header = "\n".join(lines)
    segments: list[str] = []
    for chunk in chunks:
        chunk_ids = chunk.metadata.get("chunk_ids")
        if isinstance(chunk_ids, (list, tuple)) and chunk_ids:
            ids = ", ".join(str(value) for value in chunk_ids if value)
            segments.append(f"chunk_ids: {ids}\n{chunk.text.strip()}")
        elif chunk.chunk_id:
            segments.append(f"chunk_id: {chunk.chunk_id}\n{chunk.text.strip()}")
        else:
            segments.append(chunk.text.strip())

    body = "\n\n[...]\n\n".join(segment for segment in segments if segment)
    if header and body:
        return f"{header}\n\n{body}"
    return header or body


def format_context(chunks: list[ContextChunk], max_chars: int = 0) -> str:
    """Собрать фрагменты в текст, соблюдая ограничение длины."""
    blocks: list[str] = []
    current_length = 0

    for doc_id, group in _document_groups(chunks):
        merged = _merge_document_group(group)
        block = _format_document(doc_id, merged)
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
        selected = sorted(selected, key=lambda chunk: chunk.score, reverse=True)
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
        extra = sorted(extra, key=lambda chunk: chunk.score, reverse=True)
        block = "### Прочие фрагменты\n" + format_context(extra)
        separator = 2 if blocks else 0
        if max_chars <= 0 or current_length + separator + len(block) <= max_chars:
            blocks.append(block)

    return "\n\n".join(blocks)
