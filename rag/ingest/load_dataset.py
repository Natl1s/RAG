from __future__ import annotations

import logging
import pandas as pd
# from typing import Any
from rag.ingest.schema import Document, Chunk
# from rag.ingest.preprocess import normalize_ruslawod_record

logger = logging.getLogger(__name__)

def read_docs(documents):
    return [
        Document(doc_id=title, text=text, metadata={"head_doc": head})
        for title, text, head in documents
    ]

def read_csv(documents: pd.DataFrame) -> list[Chunk]:
    """Преобразует DataFrame с готовыми чанками по именам столбцов."""

    return [
        Chunk(
            doc_id=row["title"],
            text=row["chunk"],
            metadata={"head_doc": row["head"], "info": row["metadata"]},
            chunk_id=f"chunk_{index}"
        )
        for index, row in documents.iterrows()
    ]



# 2. настроить реранкер (инициализировать класс, склеивать информацию текста чанка и метадатнных, выводить это всё как-то)


