"""Сборка локального агента с внешним RAG.

Retriever'ы строятся снаружи и передаются в сервис готовыми. Агент не знает,
как создаётся индекс, и обращается к RAG только через ``search_chunks``.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from agents.config import Settings
from agents.logging_setup import setup_logging
from agents.rag import SearchFn
from agents.service import AgentService

__all__ = ["build_local_service"]


def build_local_service(
    retrievers: Any,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    search_fn: SearchFn | None = None,
) -> AgentService:
    """Собрать агент, подключённый к готовому RAG-индексу.

    Перед сборкой включает локальное логирование: консоль + файл
    ``logs/llm_<время запуска>.log``. Файл отключается переменной
    ``LLM_LOG_FILE=off``; итоговое решение эту функцию не вызывает.
    """
    setup_logging()
    return AgentService(
        retrievers,
        settings=settings,
        checkpointer=checkpointer,
        search_fn=search_fn,
    )
