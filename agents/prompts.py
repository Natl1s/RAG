"""Промпты агента.

На этапе каркаса это черновые формулировки: важно зафиксировать интерфейс
между узлами и промптами. Тексты будут уточняться отдельным шагом.
"""

from __future__ import annotations

__all__ = [
    "ANSWER_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "REWRITE_SYSTEM_PROMPT",
    "answer_user_prompt",
    "judge_user_prompt",
    "rewrite_user_prompt",
]

JUDGE_SYSTEM_PROMPT = (
    "Ты — судья релевантности. Определи, достаточно ли приведённых фрагментов "
    "законодательства, чтобы ответить на вопрос пользователя. "
    "Верни is_relevant=true только если в контексте есть нужные сведения. "
    "Ответ верни строго в формате JSON без пояснений вокруг: "
    '{"is_relevant": true|false, "reason": "краткое объяснение"}.'
)

REWRITE_SYSTEM_PROMPT = (
    "Ты помогаешь искать по базе российского законодательства. "
    "Переформулируй вопрос в короткий поисковый запрос на русском языке. "
    "Верни только новый запрос, без пояснений."
)

ANSWER_SYSTEM_PROMPT = (
    "Ты — юридический ассистент ПСБ. Отвечай на русском языке, "
    "только по предоставленному контексту. Если данных недостаточно, "
    "честно скажи об этом. Ссылайся на источники по doc_id."
)


def judge_user_prompt(query: str, context: str) -> str:
    """Пользовательская часть промпта для LLM-судьи."""
    return f"Вопрос:\n{query}\n\nКонтекст:\n{context}"


def rewrite_user_prompt(query: str, search_query: str, reason: str) -> str:
    """Пользовательская часть промпта для переформулировки запроса."""
    return (
        f"Исходный вопрос:\n{query}\n\n"
        f"Текущий поисковый запрос:\n{search_query}\n\n"
        f"Почему контекста недостаточно:\n{reason}"
    )


def answer_user_prompt(query: str, context: str, history: str = "") -> str:
    """Пользовательская часть промпта для генерации ответа."""
    parts = [f"Вопрос:\n{query}", f"Контекст:\n{context}"]
    if history:
        parts.insert(0, f"История диалога:\n{history}")
    return "\n\n".join(parts)
