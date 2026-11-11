from __future__ import annotations

# import argparse
import logging
import time
from collections.abc import Iterable
from pathlib import Path

from rag.config import RAGConfig
from rag.index.bm25 import BM25Retriever
from rag.index.contracts import RetrievedChunk
from rag.index.faiss_dense import DenseRetriever, build_faiss_index
from rag.index.hybrid import HybridRetriever
# from rag.ingest.load_dataset import load_documents_from_slice
from rag.ingest.schema import Chunk
from rag.log_config import setup_logging

logger = logging.getLogger(__name__)

def build_retrievers(
    chunks: Iterable[Chunk],
    config: RAGConfig,
    embedder: object,
) -> dict[str, object]:


    chunk_list = list(chunks)

    retrievers = {}

    bm25 = BM25Retriever.from_chunks(chunk_list)
    retrievers["bm25"] = bm25

    if config.use_dense:
        index, chunk_order = build_faiss_index(
            chunk_list,
            config.embedder.encode_passages,
            min_chunk_chars=config.min_chunk_chars,
        )


        embed_name = embedder.__class__.__name__
        logger.info("Dense retriever: embedder=%s, dim=%s", embed_name, index.d)
        dense = DenseRetriever(
            index=index,
            chunks=chunk_order,
            embed_passage=config.embedder.encode_passages,
            embed_query=config.embedder.encode_queries,
        )

        retrievers["dense"] = dense

    if config.use_hybrid and config.use_dense:
        retrievers["hybrid"] = HybridRetriever(
            bm25_retriever=bm25,
            dense_retriever=retrievers["dense"],
        )

    return retrievers
