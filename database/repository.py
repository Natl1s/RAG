from collections.abc import Iterable

from sqlalchemy.orm import Session, defer
from database.create_tables import ChunkModel
from sqlalchemy import select, delete


class ChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def save_chunks(self, chunks: Iterable[ChunkModel]) -> list[ChunkModel]:
        """Сохранить батч целиком и вернуть объекты с назначенными ID.

        Метод управляет транзакцией переданной сессии, как и delete_chunk:
        фиксирует все её изменения, при ошибке откатывает транзакцию.
        Для независимых операций следует использовать отдельную сессию.
        Пустой батч не затрагивает текущую транзакцию.
        """
        chunks = list(chunks)
        if not chunks:
            return chunks

        try:
            self.session.add_all(chunks)
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise

        return chunks

    def get_chunks(self, *, include_embeddings: bool = True) -> list[ChunkModel]:
        statement = select(ChunkModel)
        if not include_embeddings:
            statement = statement.options(defer(ChunkModel.embedding))
        result = self.session.scalars(statement)
        return result.all()

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[ChunkModel]:

        if top_k <= 0:
            return []
        distance = ChunkModel.embedding.cosine_distance(query_embedding)
        statement = select(ChunkModel).order_by(distance, ChunkModel.id).limit(top_k)
        return self.session.scalars(statement).all()

    def vector_search_with_scores(
        self, query_embedding: list[float], top_k: int = 5,
    ) -> list[tuple[ChunkModel, float]]:
        """Чанки и cosine similarity (чем больше, тем релевантнее)."""
        if top_k <= 0:
            return []
        distance = ChunkModel.embedding.cosine_distance(
            query_embedding
        )
        statement = (
            select(ChunkModel, (1 - distance).label("score"))
            .options(defer(ChunkModel.embedding))
            .order_by(distance, ChunkModel.id)
            .limit(top_k)
        )
        return [(chunk, float(score)) for chunk, score in self.session.execute(statement)]


    def delete_chunk(self, chunk_id: int) -> bool:
        """Удалить один чанк по его ID."""

        statement = delete(ChunkModel).where(
            ChunkModel.id == chunk_id
        )

        result = self.session.execute(statement)
        self.session.commit()

        return result.rowcount > 0
