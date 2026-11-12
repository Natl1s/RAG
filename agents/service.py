"""Публичный фасад агентского слоя.

Бэкенду и внешним скриптам не нужно знать про LangGraph, узлы и состояние.
Достаточно создать ``AgentService`` и вызвать ``answer`` или ``aanswer``.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from agents.config import Settings, get_settings
from agents.graph import build_graph
from agents.llm import ensure_llm_configured
from agents.state import AgentResult, AgentState
from rag.rag_pipeline import RAG

__all__ = ["AgentService"]

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_STATE_MODELS = [
    ("agents.state", name)
    for name in (
        "ClarificationRequest",
        "ContextChunk",
        "DialogTurn",
        "PremiseCheck",
        "QuestionPart",
        "RetrievalGradeResult",
        "SubQuery",
        "SubqueryGrade",
        "UnderstandResult",
        "VerificationResult",
    )
]


def _default_checkpointer() -> MemorySaver:
    """Создать checkpointer с явным списком моделей состояния."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=_STATE_MODELS)
    return MemorySaver(serde=serde)


class AgentService:
    """Сервис, который превращает вопрос пользователя в готовый ответ.

    Настройки LLM обязательны: без ``LLM_BASE_URL`` / ``LLM_API_KEY`` /
    ``LLM_MODEL`` конструктор падает с понятной ошибкой при старте.

    Parameters
    ----------
    rag_client:
        Готовый RAG с методом ``search_chunks``.
    settings:
        Настройки агента. Если не переданы, читаются из окружения.
    checkpointer:
        Хранилище состояния диалогов. По умолчанию ``MemorySaver``.
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
            checkpointer or _default_checkpointer(),
        )

    async def aanswer(self, thread_id: str, question: str) -> AgentResult:
        """Асинхронно ответить на вопрос.

        Parameters
        ----------
        thread_id:
            Идентификатор диалога. По нему checkpointer хранит историю.
        question:
            Вопрос пользователя. Может содержать историю в формате
            ``Пользователь: ...`` / ``Ассистент: ...``.

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

        state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "raw_input": question,
            "query": question,
            "history": [],
            "last_user_question": question,
            "standalone_query": "",
            "in_scope": True,
            "scope_reason": "",
            "is_complex": False,
            "question_parts": [],
            "premise_check": None,
            "clarification": None,
            "subqueries": [],
            "contexts": [],
            "coverage": "none",
            "missing_aspects": [],
            "retrieval_attempts": 0,
            "answer_attempts": 0,
            "total_steps": 0,
            "used_tokens": 0,
            "tried_queries": [],
            "answer_draft": "",
            "verification": None,
            "answer": "",
            "status": "refused",
        }
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 30,
        }
        result = await self._graph.ainvoke(state, config=config)

        answer = (result.get("answer") or "").strip()
        if not answer:
            answer = "Не удалось сформировать ответ."

        status = result.get("status") or "refused"
        clarification = result.get("clarification")
        return AgentResult(
            answer=answer,
            status=status,
            clarification=clarification,
        )

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
