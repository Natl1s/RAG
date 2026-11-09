from os import path
from pathlib import Path

import httpx

from rag.ingest.chunking import chunk_documents
from rag.embeddings.st import SentenceTransformerEmbeddings
from rag.cli.retrieve_cli import build_retrievers
from rag.ingest.load_dataset import read_docs
from ..script.embeddings import embeddings

class ExternalEmbeddings:
    def __init__(self, client):
        self.model_name = "external-embeddings"
        self.client = client

    def embed(self, text: str) -> list[float]:
        return embeddings(self.client, text)

    
def initialaze_rag(path_to_docs, client: httpx.Client):
    # 1. Загружаем документы
    documents = read_docs(path_to_docs)

    # 2. Разбиваем на чанки
    chunks = chunk_documents(
        documents,
        chunk_size_chars=1024,
        overlap_chars=64,
    )
    print(f"Чанков: {len(chunks)}")

    # 3. Инициализируем модель эмбеддингов
    embedder = ExternalEmbeddings(client)

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
    return retrievers

def search_chunks(query: str, retrievers):
    results = retrievers["hybrid"].retrieve(query, k=5)
    context = "\n\n".join(
        f"Источник: {result.doc_id}\n"
        f"Текст: {result.text}"
        for result in results
    )
    return context

# def main():
#     retrievers = initialaze_rag(DATA_DIR)
#     # 5. Ищем релевантные чанки
#     context = search_chunks(QUERY, retrievers)

#     print("\n=== КОНТЕКСТ ДЛЯ LLM ===\n")
#     print(context)


# if __name__ == "__main__":
#     main()