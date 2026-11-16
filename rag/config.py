import os
from dataclasses import dataclass, field
from rag.embeddings.embedders import SentenceTransformerEmbeddings, ExternalEmbeddings
from rag.rerankers.reranker import Reranker, ExternalReranker, QwenReranker
@dataclass
class RAGConfig:
    # Чанкер
    chunk_size: int = 600
    chunk_overlap: int = 20

    # Эмбеддер: модель, способ вычисления эмбеддингов и клиент удалённого API.
    embedding_model: str = "/opt/models/multilingual-e5-small" #"intfloat/multilingual-e5-small"
    embedding_type: str = "local"  # "local" or "remote"
    batch_size: int = 32
    client: object = None  # httpx.Client for remote embeddings

    # Поиск RAG: способ поиска, количество результатов и переранжирование.
    retrieval_type: str = "hybrid" # "bm25", "dense", or "hybrid"
    top_k: int = 5
    use_reranker: bool = True

    # Индексация RAG: используемые индексы и режимы поиска.
    use_faiss: bool = True
    use_dense: bool = True
    use_hybrid: bool = True
    hybrid_score_mode: str = "score"  # "rank" restores legacy fusion
    hybrid_bm25_scale: float = 8.0  # heuristic: BM25 score mapped to 0.5
    hybrid_w_bm25: float = 0.5
    hybrid_w_dense: float = 0.5

    # Фильтрация чанков при дальнейшей обработке RAG: допустимая длина текста.
    min_chunk_chars: int = 0
    max_chunk_chars: int = 10000

    # Reranker parametr
    reranker: object = None
    reranker_model: str = None #"models/Qwen3-Reranker-0.6B"
    n_reranks: int = 4

    vector_store: str = field(default_factory=lambda: os.getenv("VECTOR_STORE", "faiss"))


    # Инициализация эмбеддера по выбранному способу вычисления.
    def __post_init__(self):
        if self.vector_store not in {"faiss", "pgvector"}:
            raise ValueError(f"Unknown vector store: {self.vector_store}")
        if self.embedding_type == "local":
            self.embedder = SentenceTransformerEmbeddings(
                model_name=self.embedding_model,
                batch_size=self.batch_size,
            )

            if self.reranker_model != None:
                self.reranker = QwenReranker(
                    model_name=self.reranker_model,
                )
            else:
                self.reranker = Reranker(self)

        elif self.embedding_type == "remote":
            if self.client is None:
                raise ValueError("Client must be provided")

            self.embedder = ExternalEmbeddings(self.client)
            if self.use_reranker:
                self.reranker = ExternalReranker(self)
            else:
                self.reranker = Reranker(self)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
