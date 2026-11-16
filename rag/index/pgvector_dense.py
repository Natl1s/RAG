from rag.index.contracts import RetrievedChunk


class PgvectorRetriever:
    """Dense-поиск по готовым векторам; отдельная сессия на каждый запрос."""

    def __init__(self, embedder, session_factory):
        self._embedder = embedder
        self._session_factory = session_factory

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if not query or k <= 0:
            return []

        from database.repository import ChunkRepository

        vector = self._embedder.encode_queries([query])[0].tolist()
        with self._session_factory() as session:
            matches = ChunkRepository(session).vector_search_with_scores(vector, k)
            return [
                RetrievedChunk(
                    doc_id=chunk.doc_id,
                    chunk_id=str(chunk.id),
                    text=chunk.text,
                    metadata=chunk.metadata_,
                    score=score,
                )
                for chunk, score in matches
            ]
