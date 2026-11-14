from dataclasses import dataclass
from rag.embeddings.embedders import SentenceTransformerEmbeddings, ExternalEmbeddings
from rag.rerankers.reranker import Reranker, ExternalReranker
@dataclass
class RAGConfig:
    # Чанкер
    chunk_size: int = 600
    chunk_overlap: int = 50

    # Эмбеддер: модель, способ вычисления эмбеддингов и клиент удалённого API.
    embedding_model: str = "models/multilingual-e5-small"
    embedding_type: str = "local"  # "local" or "remote"
    batch_size: int = 32
    client: object = None  # httpx.Client for remote embeddings

    # Поиск RAG: способ поиска, количество результатов и переранжирование.
    retrieval_type: str = "hybrid" # "bm25", "dense", or "hybrid"
    top_k: int = 10
    use_reranker: bool = False

    # Индексация RAG: используемые индексы и режимы поиска.
    use_faiss: bool = True
    use_dense: bool = True
    use_hybrid: bool = True

    # Фильтрация чанков при дальнейшей обработке RAG: допустимая длина текста.
    min_chunk_chars: int = 50
    max_chunk_chars: int = 10000

    # Reranker parametr
    reranker: object = None
    n_reranks: int = 3

    # Инициализация эмбеддера по выбранному способу вычисления.
    def __post_init__(self):
        if self.embedding_type == "local":
            self.embedder = SentenceTransformerEmbeddings(
                model_name=self.embedding_model,
                batch_size=self.batch_size,
            )

            self.reranker = Reranker(self)

        elif self.embedding_type == "remote":
            if self.client is None:
                raise ValueError("Client must be provided")

            self.embedder = ExternalEmbeddings(self.client)
            self.reranker = ExternalReranker(self)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
