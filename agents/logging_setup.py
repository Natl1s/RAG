"""Локальная настройка логирования агентского слоя.

Логи пишутся одновременно в консоль и в файл: один файл на запуск, время
запуска фиксируется в имени файла и в первой строке лога. Файл по умолчанию
``logs/llm_<дата_время>.log`` в корне репозитория.

Настраивается только логгер ``agents``, поэтому в файл не попадают логи
сторонних библиотек — только запросы и ответы агента.

Отключение для итогового решения
--------------------------------
``setup_logging`` вызывается только из ``agents.bootstrap.build_local_service``,
которую backend и скрипты сабмита не используют. Дополнительно файл можно
выключить переменной окружения ``LLM_LOG_FILE=off`` (останется консоль).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

__all__ = ["setup_logging"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOGGER_NAME = "agents"
_OFF_VALUES = {"off", "0", "false", "no"}


def _resolve_log_file() -> Path | None:
    """Определить путь файла логов по переменной ``LLM_LOG_FILE``.

    Возможные значения:
    - не задана/пустая — новый файл ``logs/llm_<время запуска>.log``;
    - ``off``/``0``/``false``/``no`` — файл отключён;
    - путь — логировать в указанный файл.
    """
    raw = (os.getenv("LLM_LOG_FILE") or "").strip()
    if raw.lower() in _OFF_VALUES:
        return None
    if raw:
        return Path(raw)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return _REPO_ROOT / "logs" / f"llm_{stamp}.log"


def setup_logging() -> Path | None:
    """Включить логи агента: консоль + файл на запуск.

    Returns
    -------
    Path | None
        Путь к файлу логов или ``None``, если файл отключён либо
        логирование уже настроено в этом процессе (повторный вызов — no-op).
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return None

    level_name = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = _resolve_log_file()
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    logger.propagate = False

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=== Запуск: %s ===", started_at)
    if log_file is not None:
        logger.info("Логи этого запуска: %s", log_file)
    return log_file
