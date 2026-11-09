"""Сборка локального агента с реальным RAG.

Тяжёлые импорты RAG выполняются внутри функций, чтобы остальные модули
агента можно было импортировать и тестировать без загрузки моделей.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from agents.config import Settings
from agents.service import AgentService

__all__ = [
    "DEFAULT_DATA_DIR",
    "build_local_retrievers",
    "build_local_service",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = _REPO_ROOT / "dataset" / "data" / "RusLawOD"


def build_local_retrievers(
    limit: int = 100,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Построить retriever'ы RAG по локальным parquet-файлам."""
    from rag.rag_pipeline import initialaze_rag

    path = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    return initialaze_rag(path, limit=limit)


def build_local_service(
    limit: int = 100,
    data_dir: str | Path | None = None,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> AgentService:
    """Собрать агент, подключённый к локальному RAG."""
    retrievers = build_local_retrievers(limit=limit, data_dir=data_dir)
    return AgentService(
        retrievers,
        settings=settings,
        checkpointer=checkpointer,
    )
