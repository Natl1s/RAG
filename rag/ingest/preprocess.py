from __future__ import annotations

import re
from rag.ingest.schema import Document

def clean_text(text: str) -> str:
    """
    Очистка текста от HTML, лишних URL-хвостов и нормализация пробелов.

    Parameters
    ----------
    text : str
        Исходный текст

    Returns
    -------
    str
        Очищенный текст
    """
    if not text or text.isspace():
        return ""

    cleaned = text
    # Удаляем блоки table/img и любые теги
    cleaned = re.sub(r"<table.*?>.*?</table>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<img.*?>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Удаляем хвосты вида ?pict... до пробела/кавычки
    cleaned = re.sub(r"\?pict[^\s'\"]+", " ", cleaned, flags=re.IGNORECASE)

    # Нормализуем пробелы и переносы
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned
