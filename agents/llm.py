"""Доступ к LLM через OpenAI-совместимый API.

Провайдер, модель и ключ берутся из настроек (переменные окружения).
В коде нет фиксированного имени модели: на платформе используется то,
что передала платформа, локально — то, что указано в ``.env``.

Незаданные настройки LLM — громкая ошибка конфигурации (ValueError):
проблему видно сразу при старте. Если же запрос упал или модель вернула
пустой ответ, функции мягко деградируют до RAG-fallback и пишут
предупреждение в лог.

Каждый запрос логируется на уровне INFO: промпт целиком и сырой ответ
модели до нормализации (настройка логов — agents/logging_setup.py).
"""

from __future__ import annotations

import json
import logging
import re
import time
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
    "ensure_llm_configured",
    "generate_answer",
    "judge_relevance",
    "rewrite_query",
]

logger = logging.getLogger(__name__)

_LLM_TIMEOUT_SECONDS = 30.0
_LLM_MAX_RETRIES = 1


def _format_context(context: Any) -> str:
    """Привести RAG-контекст к тексту для промпта."""
    if not context:
        return ""
    if isinstance(context, str):
        return context.strip()

    parts: list[str] = []
    for item in context:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            doc_id, text = item
            parts.append(f"doc_id: {doc_id}\n{str(text).strip()}")
        else:
            parts.append(str(item).strip())
    return "\n\n".join(part for part in parts if part)


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


def ensure_llm_configured(settings: Settings) -> None:
    """Проверить настройки LLM; незаданные переменные — громкая ошибка."""
    missing = _missing_settings(settings)
    if missing:
        raise ValueError(
            "LLM не настроена: не заданы "
            + ", ".join(missing)
            + ". Впишите их в .env (шаблон — .env.example, описание — agents/README.md)."
        )


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
    label: str = "llm",
) -> str:
    """Отправить запрос в LLM и вернуть текст ответа.

    Незаданные настройки — ValueError: проблему конфигурации видно сразу.
    Ошибка API, сбой сети или пустой ответ обрабатываются мягко: пустая
    строка, вызывающий код сам выбирает безопасный fallback.

    Промпт и сырой ответ модели логируются на уровне INFO.
    """
    ensure_llm_configured(settings)

    logger.info(
        "[%s] LLM запрос: model=%s base_url=%s",
        label,
        settings.llm_model,
        settings.llm_base_url,
    )
    for message in messages:
        logger.info("[%s] промпт [%s]:\n%s", label, message.type, message.content)

    try:
        llm = _build_llm(settings, temperature=temperature, max_tokens=max_tokens)
    except Exception as exc:
        logger.warning(
            "Не удалось создать LLM-клиент: %s: %s. Используется RAG-fallback.",
            type(exc).__name__,
            exc,
        )
        return ""

    started_at = time.monotonic()
    try:
        response = await llm.ainvoke(messages)
    except Exception as exc:
        logger.warning(
            "[%s] LLM request failed: %s: %s. Используется RAG-fallback.",
            label,
            type(exc).__name__,
            exc,
        )
        return ""

    logger.info(
        "[%s] LLM ответ (сырой):\n%s",
        label,
        getattr(response, "content", ""),
    )
    logger.info("[%s] LLM вызов завершён за %.2f c", label, time.monotonic() - started_at)

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
    context: str | list[tuple[str, str]] | None,
    *,
    settings: Settings,
) -> GradeResult:
    """Оценить, достаточно ли контекста для ответа."""
    context_text = _format_context(context)
    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=judge_user_prompt(query, context_text)),
    ]
    raw = await _chat_completion(
        settings=settings,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
        label="judge",
    )

    parsed = _parse_grade(raw)
    if parsed is not None:
        return parsed

    if raw:
        logger.warning("LLM-судья вернула ответ не в JSON-формате. Используется эвристика.")

    return GradeResult(
        is_relevant=bool(context_text),
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
        label="rewrite",
    )

    cleaned = _clean_rewrite(rewritten)
    return cleaned or base


async def generate_answer(
    query: str,
    context: str | list[tuple[str, str]] | None,
    history: str = "",
    *,
    settings: Settings,
) -> str:
    """Сгенерировать ответ по RAG-контексту.

    Если LLM недоступна, возвращает сам RAG-контекст.
    """
    normalized_context = _format_context(context)
    if not normalized_context:
        return f"По запросу «{query}» ничего не найдено."

    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=answer_user_prompt(query, normalized_context, history)),
    ]
    answer = await _chat_completion(
        settings=settings,
        messages=messages,
        label="generate",
    )
    return answer or normalized_context
