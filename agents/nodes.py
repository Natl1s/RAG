"""Узлы агентского графа.

Узлы — обычные функции: получают состояние и возвращают только изменённые
поля. Вся тяжёлая работа делегируется ``RagClient`` и функциям из ``llm.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage

from agents.config import Settings
from agents.llm import generate_answer, judge_relevance, rewrite_query
from agents.prompts import answer_user_prompt, judge_user_prompt, rewrite_user_prompt
from agents.rag import RagClient
from agents.state import AgentState

__all__ = [
    "make_generate_node",
    "make_grade_node",
    "make_retrieve_node",
    "make_rewrite_node",
]

Node = Callable[[AgentState], Awaitable[dict[str, Any]]]

_ROLE_LABELS = {
    "human": "Пользователь",
    "ai": "Ассистент",
    "system": "Система",
}


def _message_text(message: AnyMessage) -> str:
    """Достать текст сообщения для истории диалога."""
    content = message.content
    return content if isinstance(content, str) else str(content)


def _format_history(messages: list[AnyMessage], limit: int) -> str:
    """Собрать последние сообщения в строку для промпта.

    Текущий вопрос пользователя уже передаётся отдельно в ``query``,
    поэтому последнее human-сообщение из истории исключается.
    """
    if not messages:
        return ""

    previous = messages[:-1] if messages[-1].type == "human" else messages
    selected = previous[-limit:]
    lines = [
        f"{_ROLE_LABELS.get(message.type, message.type)}: {_message_text(message)}"
        for message in selected
    ]
    return "\n".join(lines)


def make_retrieve_node(rag: RagClient) -> Node:
    """Создать узел поиска в RAG."""

    async def retrieve(state: AgentState) -> dict[str, Any]:
        query = state.get("search_query") or state.get("query") or ""
        context = await asyncio.to_thread(rag.search, query)
        return {"context": context}

    return retrieve


def make_grade_node(settings: Settings) -> Node:
    """Создать узел оценки релевантности контекста."""

    async def grade(state: AgentState) -> dict[str, Any]:
        result = await judge_relevance(
            state.get("query") or "",
            state.get("context") or "",
            settings=settings,
        )
        return {
            "is_relevant": result.is_relevant,
            "relevance_reason": result.reason,
        }

    return grade


def make_rewrite_node(settings: Settings) -> Node:
    """Создать узел переформулировки поискового запроса."""

    async def rewrite(state: AgentState) -> dict[str, Any]:
        query = state.get("query") or ""
        search_query = state.get("search_query") or query
        reason = state.get("relevance_reason") or ""
        new_query = await rewrite_query(
            query,
            search_query,
            reason,
            settings=settings,
        )
        return {
            "search_query": new_query.strip() or search_query,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    return rewrite


def make_generate_node(settings: Settings) -> Node:
    """Создать узел генерации финального ответа."""

    async def generate(state: AgentState) -> dict[str, Any]:
        query = state.get("query") or ""
        context = state.get("context") or ""
        history = _format_history(
            state.get("messages") or [],
            settings.history_limit,
        )
        answer = await generate_answer(
            query,
            context,
            history,
            settings=settings,
        )
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    return generate
