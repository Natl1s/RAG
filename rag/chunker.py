from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING

from rag.ingest.chunking import chunk_documents
from rag.ingest.load_dataset import read_docs
from rag.ingest.preprocess import clean_text
from rag.ingest.schema import Chunk, Document

if TYPE_CHECKING:
    from rag.config import RAGConfig


class Chunker:
    """Чтение, очистка и разбиение документов по настройкам RAG."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config

    def read_docs(self, documents: Iterable[tuple[str, str]]) -> list[Document]:
        """Прочитать пары (название, текст) и очистить тексты документов."""
        return self.clean_docs(read_docs(documents))

    def clean_docs(self, documents: Iterable[Document]) -> list[Document]:
        """Очистить тексты, сохранив исходные документы и их метаданные."""
        return [replace(doc, text=clean_text(doc.text)) for doc in documents]

    def chunk(self, documents: Iterable[Document]) -> list[Chunk]:
        """Очистить и разбить документы с размером и перекрытием из конфига."""
        return chunk_documents(
            documents,
            chunk_size_chars=self.config.chunk_size,
            overlap_chars=self.config.chunk_overlap,
        )
