"""Настройки агентского слоя.

Все значения читаются из переменных окружения, чтобы один и тот же образ
работал в локальном окружении, в Docker и на стенде. Значения по умолчанию
рассчитаны на разработку; для реального запуска задайте env-переменные.

Локальные значения можно хранить в ``.env`` в корне проекта. Файл читается
через ``python-dotenv`` с ``override=False``: реальные переменные окружения
(платформа, Docker, shell) всегда имеют приоритет над ``.env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

__all__ = ["Settings", "get_settings"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env", override=False)


def _env_str(name: str, default: str) -> str:
    """Вернуть непустую строку из env или значение по умолчанию."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _env_first(*names: str, default: str = "") -> str:
    """Вернуть первое непустое значение из набора имён env."""
    for name in names:
        value = _env_str(name, default="")
        if value:
            return value
    return default


def _env_int(name: str, default: int) -> int:
    """Прочитать целое число из env с понятной ошибкой при неверном формате."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Переменная {name} должна быть целым числом, получено: {raw!r}") from exc


def _env_int_optional(name: str) -> int | None:
    """Прочитать необязательное целое число из env."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Переменная {name} должна быть целым числом, получено: {raw!r}") from exc


def _env_float_first(*names: str, default: float) -> float:
    """Вернуть первое непустое число из набора имён env."""
    for name in names:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            continue
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"Переменная {name} должна быть числом, получено: {raw!r}") from exc
    return default


@dataclass(frozen=True)
class Settings:
    """Иммутабельная конфигурация агента.

    Parameters
    ----------
    llm_base_url:
        Базовый URL OpenAI-совместимого API. На платформе подставляется
        автоматически; локально задаётся в ``.env``. Пустое значение
        означает, что LLM недоступна и используется RAG-fallback.
    llm_api_key:
        API-ключ. Исключён из ``repr``, чтобы случайно не попасть в логи.
    llm_model:
        Имя модели для чата. Не фиксируется в коде: берётся из env.
    llm_max_tokens:
        Ограничение длины ответа. ``None`` означает значение провайдера.
    max_rewrites:
        Сколько раз разрешено переформулировать запрос.
    history_limit:
        Сколько последних сообщений передавать в промпт.
    temperature:
        Температура генерации. Для юридических ответов по умолчанию 0.
    """

    llm_base_url: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    llm_max_tokens: int | None = None
    max_rewrites: int = 2
    history_limit: int = 10
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.max_rewrites < 0:
            raise ValueError("MAX_REWRITES не может быть отрицательным")
        if self.history_limit <= 0:
            raise ValueError("HISTORY_LIMIT должен быть больше нуля")
        if self.temperature < 0:
            raise ValueError("LLM_TEMPERATURE не может быть отрицательной")
        if self.llm_max_tokens is not None and self.llm_max_tokens <= 0:
            raise ValueError("MAX_TOKENS должен быть больше нуля")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Собрать настройки из окружения (один раз на процесс)."""
    return Settings(
        llm_base_url=_env_first("LLM_BASE_URL", "OPENAI_BASE_URL", default=""),
        llm_api_key=_env_first("LLM_API_KEY", "OPENAI_API_KEY", default=""),
        llm_model=_env_first("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL", default=""),
        llm_max_tokens=_env_int_optional("MAX_TOKENS"),
        max_rewrites=_env_int("MAX_REWRITES", 2),
        history_limit=_env_int("HISTORY_LIMIT", 10),
        temperature=_env_float_first("TEMPERATURE", "LLM_TEMPERATURE", default=0.0),
    )
