#!/bin/sh
set -e

# --- Dataset (RusLawOD) ---
if [ ! -f /app/dataset/data/.download-complete ]; then
    echo "Dataset not found, downloading..."
    python /app/dataset/dataset_tool.py download
    touch /app/dataset/data/.download-complete
else
    echo "Dataset already present, skipping"
fi

# --- Embedding model ---
MODEL_DIR="${SENTENCE_TRANSFORMERS_HOME:-/app/data/cache/embeddings}"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/.model-complete" ]; then
    echo "Embedding model not found, downloading..."
    EMBEDDING_ALLOW_DOWNLOAD=true python -c "from rag.embeddings.st import SentenceTransformerEmbeddings as E; E(model_name='intfloat/multilingual-e5-small').encode_queries(['warmup'])"
    touch "$MODEL_DIR/.model-complete"
else
    echo "Embedding model already present, skipping"
fi

exec "$@"
