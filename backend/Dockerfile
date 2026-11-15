FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY rag ./rag
COPY script ./script
COPY agents ./agents

# Готовый корпус чанков: из него строится индекс на старте.
COPY dataset/result_chunks.csv ./dataset/result_chunks.csv

# Локальная модель эмбеддингов не копируется в образ: compose монтирует
# ./models в /app/models, чтобы не гонять 1.5 ГБ через build context.
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
