from database.connection import SessionLocal
from database.repository import ChunkRepository
from database.create_tables import ChunkModel


def main():
    with SessionLocal() as session:
        repository = ChunkRepository(session)

        # 1. Сохранение
        chunk = ChunkModel(
            doc_id="test_document",
            text="Испытательный срок для главного бухгалтера",
            metadata_={"article": "70"},
            embedding=[0.1] * 384,
        )
        repository.save_chunks([chunk])

        chunk_id = chunk.id
        print("Сохранён чанк:", chunk_id)

        # 2. Получение
        chunks = repository.get_chunks()
        print("Всего чанков:", len(chunks))

        # 3. Векторный поиск
        results = repository.vector_search(
            query_embedding=[0.1] * 384,
            top_k=5,
        )

        for result in results:
            print(result.id, result.doc_id, result.text)

        # 4. Удаление по ID
        deleted = repository.delete_chunk(chunk_id)
        print("Чанк удалён:", deleted)

        # 5. Проверка удаления
        remaining_ids = {
            chunk.id for chunk in repository.get_chunks()
        }

        assert chunk_id not in remaining_ids, "Чанк не удалился!"

        print("Проверка удаления пройдена!")


if __name__ == "__main__":
    main()
