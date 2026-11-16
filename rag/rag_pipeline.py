# from os import path

import threading

import httpx
import numpy as np
import pandas as pd
from pathlib import Path
from rag.retrieve import build_retrievers
# from script.embeddings import embeddings
from rag.config import RAGConfig
from rag.chunker import Chunker
from rag.index.contracts import RetrievedChunk


class RAG:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.chunker = Chunker(config)
        self.embedder = config.embedder
        self.reranker = config.reranker
        self.retrievers = {}
        self._search_lock = threading.Lock()

    def build(self, documents=None):
        """Подготовка RAG к поиску."""

        if self.config.vector_store == "pgvector":
            from database.connection import SessionLocal
            from database.repository import ChunkRepository
            from rag.ingest.schema import Chunk

            with SessionLocal() as session:
                chunks = [
                    Chunk(doc_id=c.doc_id, chunk_id=str(c.id), text=c.text, metadata=c.metadata_)
                    for c in ChunkRepository(session).get_chunks(include_embeddings=False)
                ]
            if not chunks:
                raise ValueError("Таблица chunks пуста: сначала загрузите чанки в БД")

        elif isinstance(documents, pd.DataFrame):
            chunks = self.chunker.read_csv(documents)

        elif (
            isinstance(documents, list)
            and all(
                isinstance(doc, (list, tuple))
                and len(doc) == 3
                and all(isinstance(item, str) for item in doc)
                for doc in documents
            )
        ):
            docs = self.chunker.read_docs(documents)
            chunks = self.chunker.chunk(docs)

        else:
            raise TypeError(
                f"Неподдерживаемый тип данных: {type(documents)}"
            )

        print(f"Чанков: {len(chunks)}")

        self.retrievers = build_retrievers(
            chunks,
            self.config,
            self.embedder,
        )
            

    def search_chunks(self, query: str) -> list[RetrievedChunk]:
        with self._search_lock:
            results = self.retrievers[
                self.config.retrieval_type
            ].retrieve(query, k=self.config.top_k)
            if results and self.config.use_reranker==True:
                results = self.do_rerank(query, results)
            # print(results[0])
            return results

    def format_chunk(self, chunk: RetrievedChunk) -> str:
        return (
            f"Шапка документа:\n{chunk.metadata.get('head_doc', '')}\n\n"
            f"Метаданные:\n{chunk.metadata.get('info', '')}\n\n"
            f"Текст:\n{chunk.text}"
        )

    def do_rerank(
        self, query: str, top_result: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not top_result:
            return []
        texts = [self.format_chunk(chunk) for chunk in top_result]

        # Реранкер возвращает (индекс, score)
        rankings = self.reranker.rerank(query, texts)

        # Возвращаем исходные RetrievedChunk в новом порядке
        return [top_result[index] for index in rankings]


def main():
    rag = RAG(RAGConfig())
    docs = "./dataset/result_chunks.csv"
    docs = pd.read_csv(docs)
    rag.build(docs)

    QUERY = "Какая арендная плата по договору с ООО «Северный Склад» и до какого числа её нужно вносить?"
    result = rag.search_chunks(QUERY)
    # result = rag.do_rerank(QUERY, result)

    print(result[0])
    # docks = "/home/bibeda/psb-rag/result_chunks.csv"
    # docs = pd.read_csv(docks)

    # print(docs[docs["chunk"].str.len() <= 50])
    # print(docs.shape)

if __name__ == "__main__":
    main()

