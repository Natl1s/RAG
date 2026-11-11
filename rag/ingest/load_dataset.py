from __future__ import annotations

import logging
# from typing import Any
from rag.ingest.schema import Document
# from rag.ingest.preprocess import normalize_ruslawod_record

logger = logging.getLogger(__name__)

def read_docs(documents):
    return [
        Document(doc_id=name, title=name, text=text)
        for name, text in documents
    ]
