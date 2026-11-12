"""Оценка и учёт токенов без внешних токенизаторов.

Провайдер не всегда возвращает usage, поэтому для бюджета используется
консервативная оценка по длине текста. Если usage пришёл в ответе LLM,
приоритет отдаётся фактическим значениям.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "TokenUsage",
    "estimate_messages_tokens",
    "estimate_tokens",
    "extract_usage",
    "sum_usage",
]

_CHARS_PER_TOKEN = 1.7


@dataclass(frozen=True)
class TokenUsage:
    """Расход токенов одного LLM-вызова."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    skipped: bool = False


def estimate_tokens(text: str) -> int:
    """Оценить число токенов по длине текста.

    Для русского текста коэффициент примерно 1.7 символа на токен.
    Оценка намеренно не занижает расход.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return str(content)


def estimate_messages_tokens(messages: Iterable[Any]) -> int:
    """Оценить суммарный размер промпта в токенах."""
    return sum(estimate_tokens(_message_text(message)) for message in messages)


def _usage_from_metadata(response: Any) -> TokenUsage | None:
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        input_tokens = int(usage_metadata.get("input_tokens") or 0)
        output_tokens = int(usage_metadata.get("output_tokens") or 0)
        total_tokens = int(usage_metadata.get("total_tokens") or 0)
        if input_tokens or output_tokens or total_tokens:
            return TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens or input_tokens + output_tokens,
            )

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            input_tokens = int(token_usage.get("prompt_tokens") or 0)
            output_tokens = int(token_usage.get("completion_tokens") or 0)
            total_tokens = int(token_usage.get("total_tokens") or 0)
            if input_tokens or output_tokens or total_tokens:
                return TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens or input_tokens + output_tokens,
                )

    return None


def extract_usage(
    response: Any,
    messages: Iterable[Any],
    output: str,
) -> TokenUsage:
    """Взять фактический usage или оценить его по промпту и ответу."""
    actual = _usage_from_metadata(response)
    if actual is not None:
        return actual

    input_tokens = estimate_messages_tokens(messages)
    output_tokens = estimate_tokens(output)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def sum_usage(usages: Iterable[TokenUsage]) -> int:
    """Суммарный расход токенов по списку вызовов."""
    return sum(usage.total_tokens for usage in usages)
