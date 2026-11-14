from __future__ import annotations

import logging
import math
from dataclasses import replace

from rag.index.contracts import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


def _normalize_scores(chunks: list[RetrievedChunk]) -> dict[str, float]:
    """
    Нормализация score по рангу: лучший = 1.0, худший = 0.0.

    Parameters
    ----------
    chunks : List[RetrievedChunk]
        Список чанков с исходными score

    Returns
    -------
    Dict[str, float]
        Словарь chunk_id -> нормализованный score
    """
    if not chunks:
        return {}
    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
    max_rank = len(sorted_chunks) - 1
    if max_rank == 0:
        return {c.chunk_id: 1.0 for c in sorted_chunks}
    return {c.chunk_id: 1.0 - idx / max_rank for idx, c in enumerate(sorted_chunks)}


class HybridRetriever(Retriever):
    """
    Гибридный retriever: комбинирует BM25 и Dense результаты.
    """

    def __init__(
        self,
        bm25_retriever: Retriever,
        dense_retriever: Retriever,
        w_bm25: float = 0.5,
        w_dense: float = 0.5,
        fanout: int = 2,
        score_mode: str = "score",
        bm25_scale: float = 8.0,
    ) -> None:
        if score_mode not in {"score", "rank"}:
            raise ValueError("score_mode must be 'score' or 'rank'")
        if not math.isfinite(bm25_scale) or bm25_scale <= 0:
            raise ValueError("bm25_scale must be finite and positive")
        if fanout < 1:
            raise ValueError("fanout must be positive")
        if any(not math.isfinite(w) or w < 0 for w in (w_bm25, w_dense)):
            raise ValueError("Hybrid weights must be finite and nonnegative")
        if w_bm25 + w_dense == 0:
            raise ValueError("At least one hybrid weight must be positive")
        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        self._w_bm25 = w_bm25
        self._w_dense = w_dense
        self._fanout = fanout
        self._score_mode = score_mode
        self._bm25_scale = bm25_scale

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """
        Получить top-k чанков, объединив bm25 и dense результаты.

        Parameters
        ----------
        query : str
            Запрос пользователя
        k : int
            Сколько чанков вернуть

        Returns
        -------
        list[RetrievedChunk]
            Объединённый список чанков, отсортированный по финальному score
        """
        if not query or k <= 0:
            return []

        top_n = k * self._fanout
        bm25_results = self._bm25.retrieve(query, k=top_n)
        dense_results = self._dense.retrieve(query, k=top_n)

        logger.info(
            "Hybrid retrieve: query='%s', bm25=%s, dense=%s",
            query,
            len(bm25_results),
            len(dense_results),
        )

        if self._score_mode == "rank":
            bm25_scores = _normalize_scores(bm25_results)
            dense_scores = _normalize_scores(dense_results)
        else:
            # Fixed transforms preserve score magnitude across candidate lists.
            # BM25 scale is a tunable heuristic, not a relevance probability.
            bm25_scores = {
                c.chunk_id: max(c.score, 0.0) / (max(c.score, 0.0) + self._bm25_scale)
                for c in bm25_results
            }
            # DenseRetriever uses normalized vectors: its score is cosine similarity.
            dense_scores = {
                c.chunk_id: min(1.0, max(0.0, c.score)) for c in dense_results
            }

        merged: dict[str, RetrievedChunk] = {}
        for chunk in bm25_results + dense_results:
            merged.setdefault(chunk.chunk_id, chunk)

        scored = []
        for chunk_id, chunk in merged.items():
            score = self._w_bm25 * bm25_scores.get(
                chunk_id, 0.0
            ) + self._w_dense * dense_scores.get(chunk_id, 0.0)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [
            ch if self._score_mode == "rank" else replace(ch, score=score)
            for score, ch in scored[:k]
        ]
        return top_chunks
