from sqlalchemy.orm import Session
from database.create_tables import ChunkModel
from sqlalchemy import select, delete


class ChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def save_chunk(
        self,
        doc_id: str,
        text: str,
        metadata: dict,
        embedding: list[float],
    ) -> ChunkModel:

        chunk = ChunkModel(
            doc_id=doc_id,
            text=text,
            metadata_=metadata,
            embedding=embedding,
        )

        self.session.add(chunk)
        self.session.commit()
        self.session.refresh(chunk)

        return chunk

    def get_chunks(self) -> list[ChunkModel]:
        statement = select(ChunkModel)
        result = self.session.scalars(statement)
        return result.all()

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[ChunkModel]:

        distance = ChunkModel.embedding.cosine_distance(
            query_embedding
        )
        statement = (
            select(ChunkModel)
            .order_by(distance)
            .limit(top_k)
        )
        result = self.session.scalars(statement)
        return result.all()


    def delete_chunk(self, chunk_id: int) -> bool:
        """Удалить один чанк по его ID."""

        statement = delete(ChunkModel).where(
            ChunkModel.id == chunk_id
        )

        result = self.session.execute(statement)
        self.session.commit()

        return result.rowcount > 0