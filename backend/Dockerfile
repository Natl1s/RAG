FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SENTENCE_TRANSFORMERS_HOME=/app/data/cache/embeddings

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Скачиваем embedding-модель при сборке
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small').save('/app/models/multilingual-e5-small')"

COPY backend ./backend
COPY rag ./rag
COPY database ./database
COPY dataset/result_chunks.csv /app/dataset/result_chunks.csv

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]