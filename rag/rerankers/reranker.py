from __future__ import annotations
from script.rerank import rerank
from rag.index.contracts import RetrievedChunk

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig


class Reranker:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config

    def rerank(self, query, top_retrievals: list[RetrievalChunk]) -> list[RetrievalChunk]:
        return top_retrievals[:self.config.n_reranks]


class ExternalReranker:
    def __init__(self, config):
        self.config = config 

    def rerank(self, query, top_retrievals: list[RetrievalChunk]) -> list[RetrievalChunk]:
        return rerank(self.config.client, query, top_retrievals, self.cofing.n_reranks)