from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any
import pyarrow.parquet as pq

from rag.ingest.schema import Document
from rag.ingest.preprocess import normalize_ruslawod_record
from rag.log_config import setup_logging

logger = logging.getLogger(__name__)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Загрузка локального JSONL-файла с RusLawOD-совместимыми записями.

    Parameters
    ----------
    path : str | Path
        Путь до JSONL файла

    Returns
    -------
    List[Dict[str, Any]]
        Список записей, по одной на строку
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")

    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def document_from_record(row: dict[str, Any]) -> Document:
    """Read either a raw RusLawOD row or a normalized document."""
    if "textIPS" in row:
        return normalize_ruslawod_record(row)
    return Document(
        doc_id=str(row.get("doc_id") or "unknown"),
        title=str(row.get("title") or ""),
        text=row.get("text") or "",
        metadata=dict(row.get("metadata") or {}),
    )


def load_documents_from_slice(path: str | Path) -> list[Document]:
    """Load a raw or normalized JSONL slice, optionally gzip-compressed."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [document_from_record(json.loads(line)) for line in stream if line.strip()]


def load_documents(data_dir: Path, limit: int = 1000) -> list[Document]:
    """Load and normalize up to limit nonempty documents from Parquet."""
    docs: list[Document] = []
    if limit <= 0:
        return docs

    parquet_files = sorted(data_dir.glob("*.parquet"))

    for file in parquet_files:
        parquet_file = pq.ParquetFile(file)

        for batch in parquet_file.iter_batches(batch_size=100):
            for row in batch.to_pylist():
                text = row.get("textIPS") if "textIPS" in row else row.get("text")

                if not isinstance(text, str) or not text.strip():
                    continue
                docs.append(document_from_record(row))
                
                if len(docs) >= limit:
                    return docs

    return docs
