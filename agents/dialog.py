"""Разбор истории диалога из входного текста.

Бэкенд передаёт одну строку с репликами вида ``Пользователь: ...`` и
``Ассистент: ...``. История разбирается детерминированно, без LLM: это
дёшево и не создаёт дополнительного вызова модели.
"""

from __future__ import annotations

import re

from agents.state import DialogTurn

__all__ = ["format_history", "normalize_query", "parse_dialog", "queries_similar"]

_ROLE_RE = re.compile(
    r"(?im)^\s*(Пользователь|Ассистент|Система|User|Assistant|System)\s*:\s*"
)
_ROLE_MAP = {
    "пользователь": "user",
    "user": "user",
    "ассистент": "assistant",
    "assistant": "assistant",
    "система": "system",
    "system": "system",
}
_ROLE_LABELS = {
    "user": "Пользователь",
    "assistant": "Ассистент",
    "system": "Система",
}


def parse_dialog(raw_input: str) -> tuple[list[DialogTurn], str]:
    """Разделить вход на историю и последний вопрос пользователя.

    Если маркеров ролей нет, весь текст считается последним вопросом, а
    история остаётся пустой. Реплики до первого маркера игнорируются.
    """
    text = (raw_input or "").strip()
    if not text:
        return [], ""

    matches = list(_ROLE_RE.finditer(text))
    if not matches:
        return [], text

    turns: list[DialogTurn] = []
    for index, match in enumerate(matches):
        role = _ROLE_MAP[match.group(1).lower()]
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            turns.append(DialogTurn(role=role, text=content))

    last_user_index = next(
        (index for index in range(len(turns) - 1, -1, -1) if turns[index].role == "user"),
        None,
    )
    if last_user_index is None:
        return [], text

    return turns[:last_user_index], turns[last_user_index].text


def format_history(history: list[DialogTurn], limit: int = 10) -> str:
    """Собрать последние реплики истории в строку для промпта."""
    if not history:
        return ""

    selected = history[-limit:] if limit > 0 else history
    return "\n".join(
        f"{_ROLE_LABELS.get(turn.role, turn.role)}: {turn.text.strip()}"
        for turn in selected
        if turn.text.strip()
    )


def normalize_query(text: str) -> str:
    """Нормализовать поисковый запрос для сравнения."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def queries_similar(left: str, right: str, threshold: float = 0.9) -> bool:
    """Проверить, что два запроса почти совпадают."""
    from difflib import SequenceMatcher

    first = normalize_query(left)
    second = normalize_query(right)
    if not first or not second:
        return False
    if first == second:
        return True
    return SequenceMatcher(None, first, second).ratio() >= threshold
