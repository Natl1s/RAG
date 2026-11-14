# from os import path

import threading

import httpx
import numpy as np
import pandas as pd
from pathlib import Path
from rag.retrieve import build_retrievers
from script.embeddings import embeddings
from rag.config import RAGConfig
from rag.chunker import Chunker
from rag.index.contracts import RetrievedChunk


class RAG:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.chunker = Chunker(config)
        self.embedder = config.embedder
        self.reranker = config.reranker
        self.retrievers = {}
        self._search_lock = threading.Lock()

    def build(self, documents):
        ''' из прочитанных документов строим индексы для поиска '''


        if isinstance(documents, pd.DataFrame):
            chunks = self.chunker.read_csv(documents)
        elif (
            isinstance(documents, list)
            and all(
                isinstance(doc, (list, tuple))
                and len(doc) == 3
                and all(isinstance(item, str) for item in doc)
                for doc in documents
            )
        ):
            docs = self.chunker.read_docs(documents)
            chunks = self.chunker.chunk(docs)
        else:
            raise TypeError(f"Неподдерживаемый тип данных: {type(documents)}. Допустимо либо pd.DataFrame, либо list([str, str, str])")

        print(f"Чанков: {len(chunks)}")
        print("Эмбеддер инициализирован:", self.embedder)
        self.retrievers = build_retrievers(
            chunks,
            self.config,
            self.embedder
        )
        print("Индексы построены:", list(self.retrievers.keys()))

    def search_chunks(self, query: str) -> list[RetrievedChunk]:
        with self._search_lock:
            results = self.retrievers[
                self.config.retrieval_type
            ].retrieve(query, k=self.config.top_k)
            if self.config.use_reranker:
                results = self.do_rerank(query, results)
            # print(results[0])
            return results

    def format_chunk(self, chunk: RetrievedChunk) -> str:
        return (
            f"Шапка документа:\n{chunk.metadata.get('head_doc', '')}\n\n"
            f"Метаданные:\n{chunk.metadata.get('info', '')}\n\n"
            f"Текст:\n{chunk.text}"
        )

    def do_rerank(
        self, query: str, top_result: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        print(top_result[0])
        texts = [self.format_chunk(chunk) for chunk in top_result]

        # Реранкер возвращает (индекс, score)
        rankings = self.reranker.rerank(query, texts)

        # Возвращаем исходные RetrievedChunk в новом порядке
        return [top_result[index] for index, score in rankings]


from pathlib import Path

import pandas as pd


def merge_chunk_csvs(
    chunks_dir: str | Path,
    headers_csv: str | Path,
    output_csv: str | Path,
    *,
    document_suffix: str = ".txt",
    sep: str = ",",
) -> pd.DataFrame:
    """
    Объединить CSV с чанками и добавить шапки документов.

    Каждый CSV в chunks_dir содержит чанки одного документа.
    Название документа формируется из имени CSV + document_suffix.

    CSV с шапками должен содержать колонки title и head.

    Если хотя бы для одного документа отсутствует head,
    функция выбрасывает ValueError и не сохраняет результат.
    """
    headers_path = Path(headers_csv).resolve()
    output_path = Path(output_csv).resolve()

    chunk_paths = sorted(
        path for path in Path(chunks_dir).glob("*.csv")
        if path.is_file()
        and path.resolve() not in {headers_path, output_path}
    )

    if not chunk_paths:
        raise ValueError(f"В папке {chunks_dir} не найдены CSV с чанками")

    # Чтение шапок
    headers = pd.read_csv(
        headers_path,
        sep=sep,
        dtype=str,
        keep_default_na=False,
    )

    if not {"title", "head"}.issubset(headers.columns):
        raise ValueError(
            "CSV с шапками должен содержать колонки title и head"
        )

    # Нормализация названий
    headers["title"] = headers["title"].str.strip()
    headers["head"] = headers["head"].str.strip()

    if headers["title"].eq("").any():
        raise ValueError("В CSV с шапками есть пустые title")

    if headers["title"].duplicated().any():
        duplicates = headers.loc[
            headers["title"].duplicated(keep=False), "title"
        ].unique()

        raise ValueError(
            "В CSV с шапками есть дублирующиеся title:\n"
            + "\n".join(duplicates)
        )

    # Чтение чанков
    tables = []

    for path in chunk_paths:
        try:
            table = pd.read_csv(
                path,
                sep=sep,
                dtype=str,
                keep_default_na=False,
            )
        except pd.errors.ParserError as e:
            raise ValueError(
                f"Ошибка чтения CSV: {path}\n{e}"
            ) from e

        if "head" in table.columns:
            raise ValueError(
                f"{path}: колонка head уже существует в CSV с чанками"
            )

        table["title"] = (path.stem + document_suffix).strip()

        tables.append(table)

    # Объединение
    result = pd.concat(tables, ignore_index=True).merge(
        headers[["title", "head"]],
        on="title",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # Проверка отсутствующих шапок
    missing = (
        result["head"].isna()
        | result["head"].str.strip().eq("")
    )

    if missing.any():
        missing_titles = result.loc[
            missing, "title"
        ].unique()

        raise ValueError(
            f"Не найдены шапки для {len(missing_titles)} документов:\n"
            + "\n".join(missing_titles)
        )

    # Сохранение
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        sep=sep,
        index=False,
    )

    return result


def main():
    # merge_chunk_csvs("/home/bibeda/psb-rag/llm_index", "/home/bibeda/psb-rag/index.csv", "/home/bibeda/psb-rag/result_chunks.csv")
    rag = RAG(RAGConfig())
    docs = "/home/bibeda/psb-rag/result_chunks.csv"
    docs = pd.read_csv(docs)
    rag.build(docs)

    QUERY = "Какая арендная плата по договору с ООО «Северный Склад» и до какого числа её нужно вносить?"
    result = rag.search_chunks(QUERY)
    # result = rag.do_rerank(QUERY, result)

    print(result)
    # docks = "/home/bibeda/psb-rag/result_chunks.csv"
    # docs = pd.read_csv(docks)

    # print(docs[docs["chunk"].str.len() <= 50])
    # print(docs.shape)

if __name__ == "__main__":
    main()

