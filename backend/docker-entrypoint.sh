#!/bin/sh
set -e

# Готовый корпус чанков уже лежит в образе: /app/dataset/result_chunks.csv.
# Исходный RusLawOD нужен только для пересборки корпуса, поэтому тяжёлая
# загрузка выключена по умолчанию: `docker compose up` не требует сети
# сверх установки зависимостей при сборке.
if [ "${DOWNLOAD_DATASET:-false}" = "true" ]; then
    if [ -f /app/dataset/data/.download-complete ]; then
        echo "Dataset already present, skipping"
    else
        echo "Downloading RusLawOD..."
        python /app/dataset/dataset_tool.py download
        touch /app/dataset/data/.download-complete
    fi
fi

echo "Starting: $*"
exec "$@"
