"""Состояние агентского графа и типы обмена с бэкендом.

``AgentState`` — единый объект, который LangGraph передаёт между узлами.
Узлы возвращают только изменённые поля, а не состояние целиком.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    "Source",
    "TokenEvent",
]


class Source(TypedDict):
    """Источник, который показываем пользователю вместе с ответом."""

    doc_id: str
    chunk_id: str
    heading: str
    doc_type: str
    doc_date: str
    score: float


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
    context: str
    sources: list[Source]
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
    sources: list[Source] = field(default_factory=list)


class TokenEvent(TypedDict):
    """Очередной фрагмент ответа для SSE-события ``token``."""

    type: Literal["token"]
    text: str


class DoneEvent(TypedDict):
    """Финальное событие потока."""

    type: Literal["done"]
    answer: str
    sources: list[Source]


AgentEvent = TokenEvent | DoneEvent
