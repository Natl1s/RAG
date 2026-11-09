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
RUN sed -i 's/\r$//' /app/backend/docker-entrypoint.sh && chmod +x /app/backend/docker-entrypoint.sh
ENTRYPOINT ["/app/backend/docker-entrypoint.sh"]

WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]