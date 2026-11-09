FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SENTENCE_TRANSFORMERS_HOME=/app/data/cache/embeddings

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY rag ./rag

COPY dataset/dataset_tool.py ./dataset/dataset_tool.py
RUN if ! ls dataset/data/RusLawOD/*.parquet >/dev/null 2>&1; then \
        echo "Dataset not found, downloading..."; \
        python dataset/dataset_tool.py download; \
    else \
        echo "Dataset already present, skipping download"; \
    fi

RUN EMBEDDING_ALLOW_DOWNLOAD=true python -c \
    "from rag.embeddings.st import SentenceTransformerEmbeddings as E; \
     E(model_name='intfloat/multilingual-e5-small').encode_queries(['warmup'])"

WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]