"""Состояние агентского графа и типы обмена с бэкендом.

``AgentState`` — единый объект, который LangGraph передаёт между узлами.
Узлы возвращают только изменённые поля, а не состояние целиком.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

__all__ = [
    "AgentEvent",
    "AgentResult",
    "AgentState",
    "DoneEvent",
    "GradeResult",
    "RagContext",
    "TokenEvent",
]

RagContext = list[tuple[str, str]]


class AgentState(TypedDict, total=False):
    """Состояние графа.

    ``total=False`` означает, что узлы и входные данные могут заполнять
    только часть полей: LangGraph сам объединит частичные обновления с
    текущим состоянием.

    ``messages`` снабжено reducer'ом ``add_messages``: новый узел не
    перезаписывает историю, а добавляет сообщения к ней.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    search_query: str
    context: RagContext
    is_relevant: bool
    relevance_reason: str
    rewrite_count: int
    answer: str


class GradeResult(BaseModel):
    """Результат LLM-судьи: хватает ли контекста для ответа."""

    is_relevant: bool = Field(
        description="True, если контекста достаточно для ответа на вопрос."
    )
    reason: str = Field(
        default="",
        description="Короткое объяснение решения судьи.",
    )


@dataclass(frozen=True)
class AgentResult:
    """Итоговый ответ агента."""

    answer: str


class TokenEvent(TypedDict):
    """Очередной фрагмент ответа для SSE-события ``token``."""

    type: Literal["token"]
    text: str


class DoneEvent(TypedDict):
    """Финальное событие потока."""

    type: Literal["done"]
    answer: str


AgentEvent = TokenEvent | DoneEvent
