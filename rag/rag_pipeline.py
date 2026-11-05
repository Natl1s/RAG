from pathlib import Path

from rag.ingest.chunking import chunk_documents
from rag.embeddings.st import SentenceTransformerEmbeddings
from rag.cli.retrieve_cli import build_retrievers
from rag.ingest.load_dataset import load_documents


DATA_DIR = Path("./dataset/data/RusLawOD")
QUERY = "Какие положения налоговой политики предлагалось учесть при доработке Налогового кодекса?"


def main():
    # 1. Загружаем документы
    print(DATA_DIR.absolute())
    if not DATA_DIR.is_dir() or not any(DATA_DIR.glob("*.parquet")):
        raise FileNotFoundError(f"Parquet-файлы не найдены: {DATA_DIR}")

    documents = load_documents(DATA_DIR, limit=100)
    if not documents:
        raise ValueError("Не загружено ни одного документа. Проверьте N_DOCS и содержимое Parquet.")

    print(f"Документов: {len(documents)}")

    # 2. Разбиваем на чанки
    chunks = chunk_documents(
        documents,
        chunk_size_chars=1024,
        overlap_chars=64,
    )
    print(f"Чанков: {len(chunks)}")

    # 3. Инициализируем модель эмбеддингов
    embedder = SentenceTransformerEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        batch_size=32,
    )
    print("Эмбеддер инициализирован:", embedder.model_name)
    # 4. Строим поисковые индексы
    retrievers = build_retrievers(
        chunks,
        use_dense=True,
        use_hybrid=True,
        embedder=embedder,
        min_chunk_chars=50,
    )
    print("Индексы построены:", list(retrievers.keys()))
    # 5. Ищем релевантные чанки
    query = QUERY
    results = retrievers["hybrid"].retrieve(query, k=5)

    # 6. Формируем контекст для LLM
    context = "\n\n".join(
        f"Источник: {result.doc_id}\n"
        f"Текст: {result.text}"
        for result in results
    )

    print("\n=== КОНТЕКСТ ДЛЯ LLM ===\n")
    print(context)


if __name__ == "__main__":
    main()