"""Доступ к LLM.

Пока это заглушки: реальные вызовы OpenAI-совместимого API появятся
отдельным шагом. Узлы графа уже работают с этим интерфейсом и не зависят
от конкретной модели.
"""

from __future__ import annotations

from agents.config import Settings
from agents.state import GradeResult

__all__ = [
    "generate_answer",
    "judge_relevance",
    "rewrite_query",
]


async def judge_relevance(
    query: str,
    context: str,
    *,
    settings: Settings,
) -> GradeResult:
    """Оценить, достаточно ли контекста для ответа.

    Заглушка: всегда считает контекст релевантным.
    """
    return GradeResult(
        is_relevant=True,
        reason="Заглушка: контекст считается релевантным.",
    )


async def rewrite_query(
    query: str,
    search_query: str,
    reason: str,
    *,
    settings: Settings,
) -> str:
    """Переформулировать поисковый запрос.

    Заглушка: возвращает текущий поисковый запрос без изменений.
    """
    return search_query


async def generate_answer(
    query: str,
    context: str,
    history: str = "",
    *,
    settings: Settings,
) -> str:
    """Сгенерировать ответ по контексту.

    Заглушка LLM: возвращает переданный RAG-контекст без изменений.
    """
    answer = context.strip()
    if answer:
        return answer
    return f"По запросу «{query}» ничего не найдено."
