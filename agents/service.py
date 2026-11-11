"""Публичный фасад агентского слоя.

Бэкенду и внешним скриптам не нужно знать про LangGraph, узлы и состояние.
Достаточно создать ``AgentService`` и вызвать ``answer`` или ``aanswer``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from agents.config import Settings, get_settings
from agents.graph import build_graph
# from agents.rag import RagClient, SearchFn
from agents.llm import ensure_llm_configured
from agents.state import AgentResult
from rag.rag_pipeline import RAG
__all__ = ["AgentService"]
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class AgentService:
    """Сервис, который превращает вопрос пользователя в готовый ответ.

    Настройки LLM обязательны: без ``LLM_BASE_URL`` / ``LLM_API_KEY`` /
    ``LLM_MODEL`` конструктор падает с понятной ошибкой при старте.

    Parameters
    ----------
    rag:
        Готовые retriever'ы RAG, построенные снаружи до старта агента.
    settings:
        Настройки агента. Если не переданы, читаются из окружения.
    checkpointer:
        Хранилище состояния диалогов. По умолчанию ``MemorySaver``.
    search_fn:
        Функция поиска для подмены в тестах. По умолчанию используется
        ``rag.rag_pipeline.search_chunks``.
    """

    def __init__(
        self,
        rag_client: RAG,
        settings: Settings | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        ensure_llm_configured(self._settings)
        self._graph = build_graph(
            rag_client,
            self._settings,
            checkpointer or MemorySaver(),
        )

    async def aanswer(self, thread_id: str, question: str) -> AgentResult:
        """Асинхронно ответить на вопрос.

        Parameters
        ----------
        thread_id:
            Идентификатор диалога. По нему checkpointer хранит историю.
        question:
            Вопрос пользователя.

        Returns
        -------
        AgentResult
            Ответ агента.
        """
        if not thread_id:
            raise ValueError("thread_id не может быть пустым")

        question = (question or "").strip()
        if not question:
            raise ValueError("Вопрос не может быть пустым")

        state = {
            "messages": [HumanMessage(content=question)],
            "query": question,
            "search_query": question,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": thread_id}}
        result = await self._graph.ainvoke(state, config=config)

        answer = (result.get("answer") or "").strip()
        if not answer:
            answer = "Не удалось сформировать ответ."

        return AgentResult(answer=answer)

    def answer(self, thread_id: str, question: str) -> str:
        """Синхронная обёртка над :meth:`aanswer`.

        Удобна для скриптов. Внутри уже работающего event loop (например,
        в Jupyter или FastAPI) использовать нельзя — там нужен ``await``
        к :meth:`aanswer`.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aanswer(thread_id, question)).answer

        raise RuntimeError(
            "answer() нельзя вызывать внутри работающего event loop. "
            "Используйте: await service.aanswer(thread_id, question)."
        )
