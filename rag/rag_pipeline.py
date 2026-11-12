# from os import path

import httpx
import numpy as np
from rag.retrieve import build_retrievers
from script.embeddings import embeddings
from rag.config import RAGConfig
from rag.chunker import Chunker

class RAG:
    def __init__(self, config: RAGConfig):
        self.config = config

        self.chunker = Chunker(config)

        self.embedder = config.embedder
        self.retrievers = {}

    def build(self, documents):
        documents = self.chunker.read_docs(documents)
        chunks = self.chunker.chunk(documents)
        print(f"Чанков: {len(chunks)}")
        print("Эмбеддер инициализирован:", self.embedder)
        self.retrievers = build_retrievers(
            chunks,
            self.config,
            self.embedder
        )
        print("Индексы построены:", list(self.retrievers.keys()))

    def search_chunks(self, query: str) -> list[tuple[str, str]]:
        results = self.retrievers[
            self.config.retrieval_type
        ].retrieve(query, k=self.config.top_k)
        return [
            (str(result.doc_id), str(result.text))
            for result in results
        ]