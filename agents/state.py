"""Состояние агентского графа и типы обмена с бэкендом.

``AgentState`` — единый объект, который LangGraph передаёт между узлами.
Узлы возвращают только изменённые поля, а не состояние целиком.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

__all__ = [
    "AgentEvent",
    "AgentResult",
    "AgentState",
    "ClarificationRequest",
    "ContextChunk",
    "DialogTurn",
    "DoneEvent",
    "PremiseCheck",
    "QuestionPart",
    "RetrievalGradeResult",
    "SubQuery",
    "SubqueryGrade",
    "TokenEvent",
    "UnderstandResult",
    "VerificationResult",
]

Coverage = Literal["full", "partial", "none"]
EvidenceType = Literal["direct", "contradiction", "related", "none"]
Verdict = Literal["pass", "revise_answer", "re_retrieve", "refuse"]
FailureType = Literal[
    "none",
    "retrieval",
    "generation",
    "citation",
    "premise",
    "policy",
]


class DialogTurn(BaseModel):
    """Одна реплика истории диалога."""

    role: Literal["user", "assistant", "system"]
    text: str = ""


class QuestionPart(BaseModel):
    """Смысловая часть вопроса пользователя."""

    id: int = 1
    text: str = ""
    kind: str = "fact"


class PremiseCheck(BaseModel):
    """Проверка ложных предпосылок в вопросе."""

    has_false_premise: bool = False
    false_premise: str = ""
    explanation: str = ""


class SubQuery(BaseModel):
    """Поисковый подзапрос, порождённый декомпозицией вопроса."""

    id: int = 1
    text: str = ""
    part_ids: list[int] = Field(default_factory=list)
    kind: str = "fact"
    required: bool = True
    coverage: Coverage | None = None
    evidence_type: EvidenceType | None = None
    missing_aspects: list[str] = Field(default_factory=list)
    missing_required_aspects: list[str] = Field(default_factory=list)
    missing_optional_details: list[str] = Field(default_factory=list)
    unsupported_user_assumptions: list[str] = Field(default_factory=list)
    query_variants: list[str] = Field(default_factory=list)
    attempts: int = 0


class ContextChunk(BaseModel):
    """Нормализованный фрагмент RAG-контекста."""

    doc_id: str = ""
    chunk_id: str = ""
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    subquery_ids: list[int] = Field(default_factory=list)


class ClarificationRequest(BaseModel):
    """Запрос уточнения у пользователя."""

    needed: bool = False
    question: str = ""
    missing_slots: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)
    source: Literal["understand", "retrieval"] = "understand"
    reason: str = ""


class UnderstandResult(BaseModel):
    """Результат анализа вопроса: standalone-запрос и декомпозиция."""

    standalone_query: str = ""
    in_scope: bool = True
    scope_reason: str = ""
    is_complex: bool = False
    question_parts: list[QuestionPart] = Field(default_factory=list)
    premise_check: PremiseCheck = Field(default_factory=PremiseCheck)
    subqueries: list[SubQuery] = Field(default_factory=list)
    clarification: ClarificationRequest = Field(default_factory=ClarificationRequest)


class SubqueryGrade(BaseModel):
    """Оценка покрытия одного подзапроса."""

    subquery_id: int = 1
    coverage: Coverage = "none"
    evidence_type: EvidenceType = "none"
    missing_aspects: list[str] = Field(default_factory=list)
    missing_required_aspects: list[str] = Field(default_factory=list)
    missing_optional_details: list[str] = Field(default_factory=list)
    unsupported_user_assumptions: list[str] = Field(default_factory=list)
    reason: str = ""


class RetrievalGradeResult(BaseModel):
    """Оценка покрытия всего набора подзапросов."""

    subqueries: list[SubqueryGrade] = Field(default_factory=list)
    overall: Coverage = "none"
    evidence_type: EvidenceType = "none"
    missing_aspects: list[str] = Field(default_factory=list)
    missing_required_aspects: list[str] = Field(default_factory=list)
    missing_optional_details: list[str] = Field(default_factory=list)
    unsupported_user_assumptions: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """Результат проверки черновика ответа."""

    verdict: Verdict = "pass"
    grounded: bool = True
    answers_question: bool = True
    citation_ok: bool = True
    premise_ok: bool = True
    core_answer_supported: bool = True
    core_answer_relevant: bool = True
    premise_corrected: bool = True
    optional_details_missing: list[str] = Field(default_factory=list)
    failure_type: FailureType = "none"
    missing_evidence: list[str] = Field(default_factory=list)
    fix_hint: str = ""
    reason: str = ""


class AgentState(TypedDict, total=False):
    """Состояние графа.

    ``total=False`` означает, что узлы и входные данные могут заполнять
    только часть полей: LangGraph сам объединит частичные обновления с
    текущим состоянием.

    ``messages`` снабжено reducer'ом ``add_messages``: новый узел не
    перезаписывает историю, а добавляет сообщения к ней.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    raw_input: str
    query: str
    history: list[DialogTurn]
    last_user_question: str
    standalone_query: str
    in_scope: bool
    scope_reason: str
    is_complex: bool
    question_parts: list[QuestionPart]
    premise_check: PremiseCheck | None
    clarification: ClarificationRequest | None
    subqueries: list[SubQuery]
    contexts: list[ContextChunk]
    coverage: Coverage
    evidence_type: EvidenceType
    missing_aspects: list[str]
    missing_required_aspects: list[str]
    missing_optional_details: list[str]
    unsupported_user_assumptions: list[str]
    retrieval_attempts: int
    answer_attempts: int
    total_steps: int
    used_tokens: int
    tried_queries: list[str]
    answer_draft: str
    verification: VerificationResult | None
    answer: str
    status: Literal["answered", "partial", "refused", "clarification"]


@dataclass(frozen=True)
class AgentResult:
    """Итоговый ответ агента."""

    answer: str
    status: Literal["answered", "partial", "refused", "clarification"] = "answered"
    clarification: ClarificationRequest | None = None


class TokenEvent(TypedDict):
    """Очередной фрагмент ответа для SSE-события ``token``."""

    type: Literal["token"]
    text: str


class DoneEvent(TypedDict):
    """Финальное событие потока."""

    type: Literal["done"]
    answer: str


AgentEvent = TokenEvent | DoneEvent
