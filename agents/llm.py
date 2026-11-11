"""Доступ к LLM через OpenAI-совместимый API.

Провайдер, модель и ключ берутся из настроек (переменные окружения).
В коде нет фиксированного имени модели: на платформе используется то,
что передала платформа, локально — то, что указано в ``.env``.

Если LLM недоступна (нет ключа/URL/модели, ошибка сети или API), функции
безопасно деградируют до RAG-fallback и пишут предупреждение в консоль.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.config import Settings
from agents.prompts import (
    ANSWER_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    answer_user_prompt,
    judge_user_prompt,
    rewrite_user_prompt,
)
from agents.state import GradeResult

__all__ = [
    "generate_answer",
    "judge_relevance",
    "rewrite_query",
]

logger = logging.getLogger(__name__)

_LLM_TIMEOUT_SECONDS = 30.0
_LLM_MAX_RETRIES = 1


def _normalize_text(value: object) -> str:
    """Привести содержимое ответа модели к строке."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text.strip())
            elif isinstance(item, str):
                parts.append(item.strip())
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text.strip())
        return " ".join(part for part in parts if part).strip()
    return str(value).strip()


def _missing_settings(settings: Settings) -> list[str]:
    """Перечислить незаданные параметры, без которых LLM не вызвать."""
    missing: list[str] = []
    if not (settings.llm_base_url or "").strip():
        missing.append("LLM_BASE_URL")
    if not (settings.llm_api_key or "").strip():
        missing.append("LLM_API_KEY")
    if not (settings.llm_model or "").strip():
        missing.append("LLM_MODEL")
    return missing


def _build_llm(
    settings: Settings,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """Создать клиент ChatOpenAI по настройкам окружения."""
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "base_url": settings.llm_base_url,
        "temperature": settings.temperature if temperature is None else temperature,
        "timeout": _LLM_TIMEOUT_SECONDS,
        "max_retries": _LLM_MAX_RETRIES,
    }
    tokens = settings.llm_max_tokens if max_tokens is None else max_tokens
    if tokens is not None:
        kwargs["max_tokens"] = tokens
    return ChatOpenAI(**kwargs)


async def _chat_completion(
    *,
    settings: Settings,
    messages: list[BaseMessage],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Отправить запрос в LLM и вернуть текст ответа.

    При отсутствии настроек или ошибке API возвращает пустую строку.
    Вызывающий код сам выбирает безопасный fallback.
    """
    missing = _missing_settings(settings)
    if missing:
        logger.warning(
            "LLM недоступна: не заданы %s. Используется RAG-fallback.",
            ", ".join(missing),
        )
        return ""

    llm = _build_llm(settings, temperature=temperature, max_tokens=max_tokens)
    try:
        response = await llm.ainvoke(messages)
    except Exception as exc:
        logger.warning(
            "LLM request failed: %s: %s. Используется RAG-fallback.",
            type(exc).__name__,
            exc,
        )
        return ""

    content = _normalize_text(getattr(response, "content", ""))
    if not content:
        logger.warning("LLM вернула пустой ответ. Используется RAG-fallback.")
    return content


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Достать JSON-объект из ответа модели."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        data = json.loads(cleaned[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_grade(raw: str) -> GradeResult | None:
    """Разобрать ответ судьи, если он в ожидаемом JSON-формате."""
    data = _extract_json_object(raw)
    if not data:
        return None

    value = data.get("is_relevant")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            value = normalized == "true"
    if not isinstance(value, bool):
        return None

    reason = data.get("reason") or ""
    return GradeResult(is_relevant=value, reason=str(reason).strip())


def _clean_rewrite(value: str) -> str:
    """Убрать кавычки и служебные подписи из переформулировки."""
    cleaned = (value or "").strip()
    if not cleaned:
        return ""

    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    first_line = first_line.strip("`\"'«»")
    first_line = re.sub(
        r"^(поисковый запрос|запрос)\s*[:\-]\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    )
    return first_line.strip()


async def judge_relevance(
    query: str,
    context: str,
    *,
    settings: Settings,
) -> GradeResult:
    """Оценить, достаточно ли контекста для ответа."""
    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=judge_user_prompt(query, context)),
    ]
    raw = await _chat_completion(
        settings=settings,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
    )

    parsed = _parse_grade(raw)
    if parsed is not None:
        return parsed

    if raw:
        logger.warning("LLM-судья вернула ответ не в JSON-формате. Используется эвристика.")

    return GradeResult(
        is_relevant=bool((context or "").strip()),
        reason="LLM недоступна, использована эвристика по наличию контекста.",
    )


async def rewrite_query(
    query: str,
    search_query: str,
    reason: str,
    *,
    settings: Settings,
) -> str:
    """Переформулировать поисковый запрос.

    Если LLM недоступна, безопасно возвращает исходный запрос.
    """
    base = (search_query or query or "").strip()
    if not base:
        return ""

    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=rewrite_user_prompt(query, search_query, reason)),
    ]
    rewritten = await _chat_completion(
        settings=settings,
        messages=messages,
        temperature=0.1,
        max_tokens=128,
    )

    cleaned = _clean_rewrite(rewritten)
    return cleaned or base


async def generate_answer(
    query: str,
    context: str,
    history: str = "",
    *,
    settings: Settings,
) -> str:
    """Сгенерировать ответ по RAG-контексту.

    Если LLM недоступна, возвращает сам RAG-контекст.
    """
    normalized_context = (context or "").strip()
    if not normalized_context:
        return f"По запросу «{query}» ничего не найдено."

    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=answer_user_prompt(query, normalized_context, history)),
    ]
    answer = await _chat_completion(settings=settings, messages=messages)
    return answer or normalized_context
