from __future__ import annotations

# import argparse
import logging
import time
from collections.abc import Iterable
from pathlib import Path

from rag.config import RAGConfig
from rag.index.bm25 import BM25Retriever
from rag.index.contracts import RetrievedChunk
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
        if config.vector_store == "pgvector":
            from database.connection import SessionLocal
            from rag.index.pgvector_dense import PgvectorRetriever

            dense = PgvectorRetriever(embedder, SessionLocal)
        else:
            dense = _build_faiss_retriever(chunk_list, config, embedder)

        retrievers["dense"] = dense

    if config.use_hybrid and config.use_dense:
        retrievers["hybrid"] = HybridRetriever(
            bm25_retriever=bm25,
            dense_retriever=retrievers["dense"],
            score_mode=config.hybrid_score_mode,
            bm25_scale=config.hybrid_bm25_scale,
            w_bm25=config.hybrid_w_bm25,
            w_dense=config.hybrid_w_dense,
        )

    return retrievers


def _build_faiss_retriever(chunk_list, config, embedder):
    from rag.index.faiss_dense import DenseRetriever, build_faiss_index

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

    return dense
