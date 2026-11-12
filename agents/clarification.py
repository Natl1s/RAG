"""Определение неоднозначности по найденным документам."""

from __future__ import annotations

import re
from dataclasses import dataclass

from agents.context import metadata_hints
from agents.dialog import format_history, normalize_query
from agents.state import ClarificationRequest, ContextChunk, DialogTurn

__all__ = [
    "DocumentCandidate",
    "candidate_documents",
    "detect_context_ambiguity",
    "last_assistant_asked_question",
]

_DOCUMENT_WORDS = (
    "договор",
    "документ",
    "контракт",
    "соглашен",
    "поставк",
    "аренд",
    "гарант",
    "оплат",
    "претенз",
    "оригинал",
    "архив",
)

_ENDINGS = (
    "иями",
    "ями",
    "ами",
    "ого",
    "его",
    "ому",
    "ему",
    "ыми",
    "ими",
    "иях",
    "ях",
    "ах",
    "ов",
    "ев",
    "ий",
    "ый",
    "ая",
    "ое",
    "ые",
    "ие",
    "ой",
    "ей",
    "ом",
    "ем",
    "ую",
    "юю",
    "а",
    "я",
    "ы",
    "и",
    "у",
    "ю",
    "е",
    "о",
)

_DOC_NUMBER_RE = re.compile(
    r"[А-ЯA-Z]{1,4}[-/]\d{2,4}[-/]\d{2,4}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DocumentCandidate:
    """Документ, который может быть предметом вопроса."""

    doc_id: str
    doc_number: str
    name: str
    score: float

    @property
    def label(self) -> str:
        if self.name and self.doc_number:
            return f"{self.name} ({self.doc_number})"
        return self.name or self.doc_number


def _stem(token: str) -> str:
    if len(token) <= 3 or any(char.isdigit() for char in token):
        return token
    for ending in _ENDINGS:
        if token.endswith(ending) and len(token) - len(ending) >= 3:
            return token[: -len(ending)]
    return token


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_query(text), flags=re.UNICODE)


def _normalize_doc_number(value: str) -> str:
    return re.sub(r"\s+", "", (value or "").strip()).replace("/", "-").lower()


def _extract_name(title: str) -> str:
    match = _DOC_NUMBER_RE.search(title or "")
    if match is None:
        return ""
    tail = title[match.end() :]
    tail = re.sub(r"\.(txt|md|docx?|pdf)$", "", tail, flags=re.IGNORECASE)
    return tail.replace("_", " ").strip(" -—–:.,")


def candidate_documents(contexts: list[ContextChunk]) -> list[DocumentCandidate]:
    """Собрать уникальные документы с номерами договоров."""
    merged: dict[str, DocumentCandidate] = {}

    for chunk in contexts:
        hints = metadata_hints(chunk)
        doc_number = (hints.get("doc_number") or "").strip()
        if not doc_number:
            continue

        key = _normalize_doc_number(doc_number)
        title = str(chunk.metadata.get("title") or chunk.doc_id or "")
        name = _extract_name(title)
        existing = merged.get(key)

        if existing is None:
            merged[key] = DocumentCandidate(
                doc_id=chunk.doc_id,
                doc_number=doc_number,
                name=name,
                score=chunk.score,
            )
            continue

        merged[key] = DocumentCandidate(
            doc_id=existing.doc_id,
            doc_number=existing.doc_number,
            name=existing.name or name,
            score=max(existing.score, chunk.score),
        )

    return sorted(merged.values(), key=lambda item: item.score, reverse=True)


def _has_doc_number_selector(text: str, candidates: list[DocumentCandidate]) -> bool:
    found = {
        _normalize_doc_number(match.group(0))
        for match in _DOC_NUMBER_RE.finditer(text or "")
    }
    return any(
        _normalize_doc_number(candidate.doc_number) in found
        for candidate in candidates
    )


def _has_name_selector(text: str, candidates: list[DocumentCandidate]) -> bool:
    text_stems = {_stem(token) for token in _tokens(text)}
    for candidate in candidates:
        name_tokens = [
            token
            for token in _tokens(candidate.name)
            if len(token) >= 3
        ]
        if name_tokens and all(_stem(token) in text_stems for token in name_tokens):
            return True
    return False


def _has_selector(
    text: str,
    candidates: list[DocumentCandidate],
) -> bool:
    return _has_doc_number_selector(
        text, candidates
    ) or _has_name_selector(text, candidates)


def _looks_like_document_question(question: str) -> bool:
    normalized = normalize_query(question)
    return any(word in normalized for word in _DOCUMENT_WORDS)


def last_assistant_asked_question(history: list[DialogTurn]) -> bool:
    """Проверить, был ли последний ход ассистента уточняющим вопросом."""
    for turn in reversed(history):
        if turn.role == "assistant":
            return turn.text.strip().endswith("?")
        if turn.role == "user":
            return False
    return False


def detect_context_ambiguity(
    question: str,
    history: list[DialogTurn],
    contexts: list[ContextChunk],
) -> ClarificationRequest | None:
    """Вернуть уточнение, если найдено несколько конкурирующих документов."""
    candidates = candidate_documents(contexts)
    if len(candidates) < 2:
        return None
    if not _looks_like_document_question(question):
        return None
    if last_assistant_asked_question(history):
        return None

    history_text = format_history(history, 20)
    selector_text = f"{question}\n{history_text}"
    if _has_selector(selector_text, candidates):
        return None

    options: list[str] = []
    for candidate in candidates:
        if candidate.label and candidate.label not in options:
            options.append(candidate.label)
    if len(options) < 2:
        return None

    options = options[:4]
    return ClarificationRequest(
        needed=True,
        question=(
            "Уточните, пожалуйста, какой документ вы имеете в виду: "
            + "; ".join(options)
            + "?"
        ),
        options=options,
        source="retrieval",
        reason="Найдено несколько подходящих документов.",
    )
