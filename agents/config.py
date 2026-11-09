"""Настройки агентского слоя.

Все значения читаются из переменных окружения, чтобы один и тот же образ
работал в локальном окружении, в Docker и на стенде. Значения по умолчанию
рассчитаны на разработку; для реального запуска задайте env-переменные.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

__all__ = ["Settings", "get_settings"]


def _env_str(name: str, default: str) -> str:
    """Вернуть непустую строку из env или значение по умолчанию."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    """Прочитать целое число из env с понятной ошибкой при неверном формате."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Переменная {name} должна быть целым числом, получено: {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    """Прочитать число с плавающей точкой из env."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Переменная {name} должна быть числом, получено: {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Иммутабельная конфигурация агента.

    Parameters
    ----------
    llm_base_url:
        Базовый URL OpenAI-совместимого API. Пустая строка означает
        стандартный endpoint OpenAI.
    llm_api_key:
        API-ключ. Исключён из ``repr``, чтобы случайно не попасть в логи.
    llm_model:
        Имя модели для чата.
    rag_k:
        Сколько чанков запрашивать у RAG на один поиск.
    max_rewrites:
        Сколько раз разрешено переформулировать запрос.
    history_limit:
        Сколько последних сообщений передавать в промпт.
    temperature:
        Температура генерации. Для юридических ответов по умолчанию 0.
    """

    llm_base_url: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = "gpt-4o-mini"
    rag_k: int = 5
    max_rewrites: int = 2
    history_limit: int = 10
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.rag_k <= 0:
            raise ValueError("RAG_K должен быть больше нуля")
        if self.max_rewrites < 0:
            raise ValueError("MAX_REWRITES не может быть отрицательным")
        if self.history_limit <= 0:
            raise ValueError("HISTORY_LIMIT должен быть больше нуля")
        if self.temperature < 0:
            raise ValueError("LLM_TEMPERATURE не может быть отрицательной")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Собрать настройки из окружения (один раз на процесс)."""
    return Settings(
        llm_base_url=_env_str("OPENAI_BASE_URL", ""),
        llm_api_key=_env_str("OPENAI_API_KEY", ""),
        llm_model=_env_str("LLM_MODEL", "gpt-4o-mini"),
        rag_k=_env_int("RAG_K", 5),
        max_rewrites=_env_int("MAX_REWRITES", 2),
        history_limit=_env_int("HISTORY_LIMIT", 10),
        temperature=_env_float("LLM_TEMPERATURE", 0.0),
    )
