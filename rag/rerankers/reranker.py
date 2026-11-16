from __future__ import annotations
from script.embeddings import rerank
from rag.index.contracts import RetrievedChunk
import numpy as np
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig


class Reranker:
    """Тривиальный реранкер-заглушка: возвращает первые n_reranks как есть."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config

    def rerank(self, query: str, texts: list[str]) -> list[int]:
        n = len(texts)
        return [i for i in range(n)]

    
class ExternalReranker:
    def __init__(self, config):
        self.config = config 

    def rerank(self, query, top_retrievals: list[str]) -> list[int]:
        return rerank(self.config.client, query, top_retrievals, self.config.n_reranks)

class QwenReranker:
    """
    Обёртка над Qwen3-Reranker для оценки релевантности документов.

    Поддерживает:
    - ленивую загрузку модели;
    - батчинг;
    - вычисление оценок релевантности;
    - ранжирование документов.
    """

    def __init__(
        self,
        model_name: str = "models/Qwen3-Reranker-0.6B",
        *,
        batch_size: int = 4,
        device: str = "cpu",
        top_k = 5
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self.top_k = top_k

    @property
    def model(self):
        """Ленивая загрузка модели с кешированием."""

        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                str(Path(self.model_name).resolve()),
                device=self.device,
                local_files_only=True,
            )

        return self._model

    def score(
        self,
        query: str,
        texts: list[str],
    ) -> np.ndarray:
        """
        Вычисление релевантности текстов запросу.

        Returns
        -------
        np.ndarray
            Оценки релевантности для каждого текста.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        pairs = [(query, text) for text in texts]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def rerank(
        self,
        query: str,
        texts: list[str]
    ) -> list[tuple[int, float]]:
        """
        Ранжирование текстов по релевантности запросу.

        Returns
        -------
        list[tuple[int, float]]
            Список (индекс текста, оценка релевантности).
        """
        if self.top_k <= 0:
            return []

        scores = self.score(query, texts)

        indices = np.argsort(scores)[::-1][:self.top_k]

        return [
            (int(i), float(scores[i]))
            for i in indices
        ]