"""CLI локального запуска агента с RAG."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agents.bootstrap import build_local_service

__all__ = ["main", "parse_args"]


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Запуск агента с локальным RAG-индексом.",
    )
    parser.add_argument("thread_id", help="Идентификатор диалога")
    parser.add_argument("question", help="Вопрос пользователя")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Сколько документов загрузить в RAG (по умолчанию 100)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Путь к каталогу с parquet-файлами RusLawOD",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    service = await asyncio.to_thread(
        build_local_service,
        limit=args.limit,
        data_dir=args.data_dir,
    )
    result = await service.aanswer(args.thread_id, args.question)

    print(result.answer)
    if not result.sources:
        print("\nИсточники: не найдены")
        return

    print("\nИсточники:")
    for source in result.sources:
        print(
            f"- doc_id={source['doc_id']} chunk_id={source['chunk_id']} "
            f"heading={source['heading']!r} doc_type={source['doc_type']!r} "
            f"doc_date={source['doc_date']!r} score={source['score']:.4f}"
        )


def main() -> None:
    """Точка входа CLI."""
    asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    main()
