#!/bin/sh
set -e

MODEL_DIR="/app/data/cache/embeddings/multilingual-e5-small"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Embedding model not found, downloading..."

    mkdir -p "$MODEL_DIR"

    python -c "
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-small')
model.save('$MODEL_DIR')
"

    echo "Embedding model downloaded"
else
    echo "Embedding model already exists"
fi

exec "$@"
