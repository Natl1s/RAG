"""Узлы агентского графа.

Узлы — обычные функции: получают состояние и возвращают только изменённые
поля. Вся тяжёлая работа делегируется RAG и функциям из ``llm.py``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage

from agents.clarification import (
    detect_context_ambiguity,
    last_assistant_asked_question,
)
from agents.config import Settings
from agents.context import merge_contexts, normalize_chunk
from agents.dialog import parse_dialog, queries_similar
from agents.llm import (
    clarification_answer,
    fallback_grade,
    fallback_verification,
    generate_answer,
    grade_retrieval,
    no_data_answer,
    out_of_scope_answer,
    revise_answer,
    rewrite_subquery,
    understand_query,
    verify_answer,
)
from agents.state import AgentState, SubQuery, VerificationResult
from agents.tokens import TokenUsage, sum_usage
from rag.rag_pipeline import RAG

__all__ = [
    "make_clarify_node",
    "make_finalize_node",
    "make_generate_node",
    "make_grade_node",
    "make_parse_dialog_node",
    "make_refuse_node",
    "make_retrieve_node",
    "make_revise_node",
    "make_rewrite_node",
    "make_understand_node",
    "make_verify_node",
    "route_after_grade",
    "route_after_understand",
    "route_after_verify",
]

logger = logging.getLogger(__name__)

Node = Callable[[AgentState], Awaitable[dict[str, Any]]]


def _question(state: AgentState) -> str:
    return (
        state.get("standalone_query")
        or state.get("last_user_question")
        or state.get("query")
        or ""
    ).strip()


def _budget_remaining(state: AgentState, settings: Settings) -> int:
    """Сколько токенов ещё можно потратить на LLM в этом вопросе."""
    used = state.get("used_tokens", 0)
    return max(0, settings.max_total_tokens_per_question - used)


def _add_usage(state: AgentState, usage: list[TokenUsage]) -> int:
    """Вернуть обновлённый суммарный расход токенов."""
    return state.get("used_tokens", 0) + sum_usage(usage)


def _subqueries(state: AgentState) -> list[SubQuery]:
    subqueries = list(state.get("subqueries") or [])
    if subqueries:
        return subqueries

    question = _question(state)
    if not question:
        return []
    return [SubQuery(id=1, text=question, part_ids=[1], kind="fact", required=True)]


def make_parse_dialog_node() -> Node:
    """Создать узел детерминированного разбора истории диалога."""

    async def parse_dialog_node(state: AgentState) -> dict[str, Any]:
        raw_input = (
            state.get("raw_input")
            or state.get("query")
            or ""
        ).strip()
        history, last_question = parse_dialog(raw_input)
        if not last_question:
            last_question = (state.get("query") or "").strip()
        return {
            "raw_input": raw_input,
            "history": history,
            "last_user_question": last_question,
            "query": last_question,
        }

    return parse_dialog_node


def make_understand_node(settings: Settings) -> Node:
    """Создать узел анализа вопроса и декомпозиции."""

    async def understand(state: AgentState) -> dict[str, Any]:
        last_question = (
            state.get("last_user_question")
            or state.get("query")
            or ""
        ).strip()
        usage: list[TokenUsage] = []
        result = await understand_query(
            state.get("history") or [],
            last_question,
            settings=settings,
            usage_sink=usage,
            budget_remaining=_budget_remaining(state, settings),
        )
        subqueries = [
            subquery.model_copy(deep=True) for subquery in result.subqueries
        ]
        clarification = result.clarification
        if clarification.needed and last_assistant_asked_question(
            state.get("history") or []
        ):
            clarification = clarification.model_copy(
                update={"needed": False, "question": ""}
            )
        return {
            "query": last_question,
            "standalone_query": result.standalone_query,
            "in_scope": result.in_scope,
            "scope_reason": result.scope_reason,
            "is_complex": result.is_complex,
            "question_parts": result.question_parts,
            "premise_check": result.premise_check,
            "clarification": clarification,
            "subqueries": subqueries,
            "contexts": [],
            "tried_queries": [],
            "retrieval_attempts": 0,
            "answer_attempts": 0,
            "total_steps": state.get("total_steps", 0) + 1,
            "used_tokens": _add_usage(state, usage),
        }

    return understand


def make_retrieve_node(rag: RAG, settings: Settings) -> Node:
    """Создать узел поиска в RAG по всем подзапросам."""

    async def retrieve(state: AgentState) -> dict[str, Any]:
        subqueries = _subqueries(state)
        tried_queries = list(state.get("tried_queries") or [])
        collected = []

        for subquery in subqueries:
            query = (subquery.text or "").strip()
            if not query:
                continue

            try:
                chunks = await asyncio.to_thread(rag.search_chunks, query)
            except Exception as exc:
                logger.warning(
                    "RAG поиск упал для подзапроса %s: %s: %s",
                    subquery.id,
                    type(exc).__name__,
                    exc,
                )
                chunks = []

            logger.info(
                "RAG поиск: подзапрос=%s, запрос=%r, найдено=%s",
                subquery.id,
                query,
                len(chunks or []),
            )
            for chunk in chunks or []:
                collected.append(normalize_chunk(chunk, subquery.id))

            if query not in tried_queries:
                tried_queries.append(query)

        return {
            "subqueries": subqueries,
            "contexts": merge_contexts(collected, settings.max_merged_chunks),
            "tried_queries": tried_queries,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    return retrieve


def make_grade_node(settings: Settings) -> Node:
    """Создать узел оценки покрытия контекста."""

    async def grade(state: AgentState) -> dict[str, Any]:
        question = _question(state)
        subqueries = _subqueries(state)
        contexts = list(state.get("contexts") or [])
        total_steps = state.get("total_steps", 0)
        usage: list[TokenUsage] = []

        if contexts and total_steps < settings.max_total_llm_calls:
            result = await grade_retrieval(
                question,
                subqueries,
                contexts,
                settings=settings,
                usage_sink=usage,
                budget_remaining=_budget_remaining(state, settings),
            )
            step_increment = 1
        elif contexts:
            result = fallback_grade(subqueries, contexts)
            step_increment = 0
        else:
            result = await grade_retrieval(
                question,
                subqueries,
                contexts,
                settings=settings,
                usage_sink=usage,
                budget_remaining=_budget_remaining(state, settings),
            )
            step_increment = 0

        by_id = {item.subquery_id: item for item in result.subqueries}
        updated: list[SubQuery] = []
        for subquery in subqueries:
            item = by_id.get(subquery.id)
            if item is None:
                updated.append(subquery)
                continue
            updated.append(
                subquery.model_copy(
                    update={
                        "coverage": item.coverage,
                        "missing_aspects": item.missing_aspects,
                    }
                )
            )

        clarification = None
        if result.overall != "none":
            clarification = detect_context_ambiguity(
                question,
                state.get("history") or [],
                contexts,
            )

        return {
            "subqueries": updated,
            "coverage": result.overall,
            "missing_aspects": result.missing_aspects,
            "clarification": clarification,
            "total_steps": total_steps + step_increment,
            "used_tokens": _add_usage(state, usage),
        }

    return grade


def make_rewrite_node(settings: Settings) -> Node:
    """Создать узел переформулировки проблемных подзапросов."""

    async def rewrite(state: AgentState) -> dict[str, Any]:
        question = _question(state)
        verification = state.get("verification")
        subqueries = [
            subquery.model_copy(deep=True)
            for subquery in _subqueries(state)
        ]
        tried_queries = list(state.get("tried_queries") or [])

        target_ids = {
            subquery.id
            for subquery in subqueries
            if subquery.coverage != "full"
        }
        if not target_ids and subqueries:
            target_ids = {subqueries[0].id}

        if verification is not None and verification.missing_evidence:
            next_id = max((subquery.id for subquery in subqueries), default=0) + 1
            for evidence in verification.missing_evidence:
                evidence = (evidence or "").strip()
                if not evidence:
                    continue
                if len(subqueries) >= settings.max_subqueries:
                    break
                if any(
                    queries_similar(evidence, subquery.text)
                    for subquery in subqueries
                ):
                    continue
                subqueries.append(
                    SubQuery(
                        id=next_id,
                        text=evidence,
                        part_ids=[],
                        kind="fact",
                        required=True,
                    )
                )
                target_ids.add(next_id)
                next_id += 1

        available = settings.max_total_llm_calls - state.get("total_steps", 0)
        rewrite_limit = min(available, settings.max_rewrite_subqueries_per_round)
        rewrite_calls = 0
        usage: list[TokenUsage] = []
        updated: list[SubQuery] = []

        for subquery in subqueries:
            if subquery.id not in target_ids or rewrite_calls >= rewrite_limit:
                updated.append(subquery)
                continue

            reason = "; ".join(subquery.missing_aspects)
            if not reason and verification is not None:
                reason = verification.fix_hint or verification.reason
            if not reason:
                reason = "Текущий контекст не покрывает подзапрос."

            new_text = await rewrite_subquery(
                question,
                subquery,
                reason,
                tried_queries,
                settings=settings,
                usage_sink=usage,
                budget_remaining=max(
                    0,
                    settings.max_total_tokens_per_question
                    - state.get("used_tokens", 0)
                    - sum_usage(usage),
                ),
            )
            rewrite_calls += 1
            new_text = (new_text or "").strip() or subquery.text

            if new_text != subquery.text:
                updated.append(
                    subquery.model_copy(
                        update={
                            "text": new_text,
                            "coverage": None,
                            "missing_aspects": [],
                            "attempts": subquery.attempts + 1,
                        }
                    )
                )
            else:
                updated.append(
                    subquery.model_copy(
                        update={"attempts": subquery.attempts + 1}
                    )
                )

            if new_text not in tried_queries:
                tried_queries.append(new_text)

        kept_contexts = [
            chunk
            for chunk in (state.get("contexts") or [])
            if not chunk.subquery_ids
            or any(subquery_id not in target_ids for subquery_id in chunk.subquery_ids)
        ]

        return {
            "subqueries": updated,
            "contexts": kept_contexts,
            "tried_queries": tried_queries,
            "total_steps": state.get("total_steps", 0) + rewrite_calls,
            "used_tokens": _add_usage(state, usage),
        }

    return rewrite


def make_generate_node(settings: Settings) -> Node:
    """Создать узел генерации черновика ответа."""

    async def generate(state: AgentState) -> dict[str, Any]:
        question = _question(state)
        contexts = list(state.get("contexts") or [])
        answer_attempts = state.get("answer_attempts", 0) + 1
        usage: list[TokenUsage] = []

        if not contexts:
            return {
                "answer_draft": no_data_answer(question),
                "answer_attempts": answer_attempts,
                "used_tokens": _add_usage(state, usage),
            }

        answer = await generate_answer(
            question,
            contexts,
            state.get("history") or [],
            state.get("premise_check"),
            state.get("coverage") or "none",
            state.get("missing_aspects") or [],
            settings=settings,
            usage_sink=usage,
            budget_remaining=_budget_remaining(state, settings),
        )
        return {
            "answer_draft": answer,
            "answer_attempts": answer_attempts,
            "total_steps": state.get("total_steps", 0) + 1,
            "used_tokens": _add_usage(state, usage),
        }

    return generate


def make_verify_node(settings: Settings) -> Node:
    """Создать узел проверки черновика ответа."""

    async def verify(state: AgentState) -> dict[str, Any]:
        question = _question(state)
        draft = (state.get("answer_draft") or "").strip()
        contexts = list(state.get("contexts") or [])

        if not draft:
            return {
                "verification": VerificationResult(
                    verdict="refuse",
                    grounded=False,
                    answers_question=False,
                    citation_ok=False,
                    failure_type="policy",
                    reason="Пустой черновик ответа.",
                )
            }

        total_steps = state.get("total_steps", 0)
        usage: list[TokenUsage] = []
        if not contexts:
            result = await verify_answer(
                question,
                draft,
                contexts,
                state.get("premise_check"),
                settings=settings,
                usage_sink=usage,
                budget_remaining=_budget_remaining(state, settings),
            )
            step_increment = 0
        elif total_steps >= settings.max_total_llm_calls:
            result = fallback_verification(draft, contexts)
            step_increment = 0
        else:
            result = await verify_answer(
                question,
                draft,
                contexts,
                state.get("premise_check"),
                settings=settings,
                usage_sink=usage,
                budget_remaining=_budget_remaining(state, settings),
            )
            step_increment = 1

        return {
            "verification": result,
            "total_steps": total_steps + step_increment,
            "used_tokens": _add_usage(state, usage),
        }

    return verify


def make_revise_node(settings: Settings) -> Node:
    """Создать узел исправления черновика ответа."""

    async def revise(state: AgentState) -> dict[str, Any]:
        question = _question(state)
        draft = (state.get("answer_draft") or "").strip()
        contexts = list(state.get("contexts") or [])
        verification = state.get("verification")

        if not draft or not contexts or verification is None:
            return {}

        usage: list[TokenUsage] = []
        revised = await revise_answer(
            question,
            draft,
            contexts,
            verification,
            settings=settings,
            usage_sink=usage,
            budget_remaining=_budget_remaining(state, settings),
        )
        return {
            "answer_draft": revised,
            "answer_attempts": state.get("answer_attempts", 0) + 1,
            "total_steps": state.get("total_steps", 0) + 1,
            "used_tokens": _add_usage(state, usage),
        }

    return revise


def make_clarify_node() -> Node:
    """Создать узел уточняющего вопроса."""

    async def clarify(state: AgentState) -> dict[str, Any]:
        answer = clarification_answer(state.get("clarification"))
        return {
            "answer": answer,
            "status": "clarification",
            "messages": [AIMessage(content=answer)],
        }

    return clarify


def make_finalize_node() -> Node:
    """Создать узел финального ответа."""

    async def finalize(state: AgentState) -> dict[str, Any]:
        question = _question(state)
        coverage = state.get("coverage") or "none"
        verification = state.get("verification")
        draft = (state.get("answer_draft") or "").strip()

        if not state.get("in_scope", True):
            answer = out_of_scope_answer()
            status = "refused"
        elif verification is not None and verification.verdict == "refuse":
            answer = no_data_answer(question)
            status = "refused"
        elif not draft or coverage == "none":
            answer = no_data_answer(question)
            status = "refused"
        else:
            answer = draft
            status = "answered"
            if coverage == "partial":
                status = "partial"
                missing = state.get("missing_aspects") or []
                if missing and "не хватает" not in answer.lower():
                    answer = (
                        f"{answer}\n\nВ базе знаний не хватает: "
                        f"{', '.join(missing)}."
                    )
            elif verification is not None and verification.verdict != "pass":
                status = "partial"

        logger.info(
            "Итог по вопросу: status=%s, LLM-вызовов=%s, токенов≈%s",
            status,
            state.get("total_steps", 0),
            state.get("used_tokens", 0),
        )
        return {
            "answer": answer,
            "status": status,
            "messages": [AIMessage(content=answer)],
        }

    return finalize


def make_refuse_node() -> Node:
    """Создать узел отказа для вопросов вне тематики."""

    async def refuse(state: AgentState) -> dict[str, Any]:
        answer = out_of_scope_answer()
        return {
            "answer": answer,
            "status": "refused",
            "messages": [AIMessage(content=answer)],
        }

    return refuse


def route_after_understand(state: AgentState) -> str:
    """Маршрут после анализа вопроса."""
    if not state.get("in_scope", True):
        return "refuse"
    clarification = state.get("clarification")
    if clarification is not None and clarification.needed:
        return "clarify"
    return "retrieve"


def route_after_grade(state: AgentState, settings: Settings) -> str:
    """Маршрут после оценки покрытия контекста."""
    coverage = state.get("coverage") or "none"
    total_steps = state.get("total_steps", 0)
    retrieval_attempts = state.get("retrieval_attempts", 0)

    clarification = state.get("clarification")
    if (
        coverage != "none"
        and clarification is not None
        and clarification.needed
    ):
        return "clarify"

    if coverage == "full":
        return "generate" if total_steps < settings.max_total_llm_calls else "finalize"

    if state.get("used_tokens", 0) >= settings.max_total_tokens_soft:
        if coverage == "partial" and total_steps < settings.max_total_llm_calls:
            return "generate"
        return "finalize"

    can_retry = (
        retrieval_attempts < settings.max_retrieval_attempts
        and total_steps < settings.max_total_llm_calls
    )
    if coverage in {"partial", "none"} and can_retry:
        return "rewrite"
    if coverage == "none":
        return "finalize"
    return "generate" if total_steps < settings.max_total_llm_calls else "finalize"


def route_after_verify(state: AgentState, settings: Settings) -> str:
    """Маршрут после проверки ответа."""
    verification = state.get("verification")
    if verification is None or verification.verdict == "pass":
        return "finalize"

    total_steps = state.get("total_steps", 0)
    if total_steps >= settings.max_total_llm_calls:
        return "finalize"
    if state.get("used_tokens", 0) >= settings.max_total_tokens_soft:
        return "finalize"

    if verification.verdict == "revise_answer":
        if state.get("answer_attempts", 0) < settings.max_answer_attempts:
            return "revise"
        return "finalize"

    if verification.verdict == "re_retrieve":
        if state.get("retrieval_attempts", 0) < settings.max_retrieval_attempts:
            return "rewrite"
        return "finalize"

    return "finalize"
