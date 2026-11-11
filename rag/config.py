from dataclasses import dataclass


@dataclass
class RAGConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200

    embedding_model: str = "models/multilingual-e5-small"

    retrieval_type: str = "hybrid"
    top_k: int = 5

    use_reranker: bool = False