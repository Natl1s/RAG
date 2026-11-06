# RAG по RusLawOD

В этой папке — загрузка юридических документов, разбиение на чанки и поиск
через BM25, Dense (FAISS) и Hybrid. `rag_pipeline.py` находит фрагменты и печатает
контекст для LLM. Саму LLM он пока не вызывает.


## Что находится в папке

| Путь | Назначение |
| --- | --- |
| `rag_pipeline.py` | Пример полного прохода от Parquet до контекста для LLM |
| `RAG_RusLaw_Demo.ipynb` | Просмотр документов и чанков, сравнение результатов трёх поисковиков |
| `ingest/` | Загрузка Parquet и JSONL, нормализация метаданных, очистка и чанкинг |
| `embeddings/` | Настройки моделей и расчёт эмбеддингов через Sentence Transformers |
| `index/` | BM25, FAISS, Hybrid и тип результата `RetrievedChunk` |
| `cli/` | Сборка поисковиков (`build_retrievers`), поиск и утилиты для JSONL-срезов |
| `ui/service.py` | Обёртки загрузки, чанкинга и поиска для интерфейса; отдельного UI здесь нет |
| `log_config.py` | Настройка логирования через `LOG_LEVEL` и `LOG_FILE` |

Примечание: JSONL-срезы не реализованы. это пережиток прошлого этой части проекта.

## Подготовка

Команды ниже выполняются **из корня `psb-rag/`**, а не из `rag/`.
Рабочее окружение проекта — Python 3.12.

1. Создать окружение, если его ещё нет, и установить зависимости:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install numpy pandas pyarrow rank-bm25 faiss-cpu sentence-transformers huggingface-hub
   ```

2. Скачать документы, если в `dataset/data/RusLawOD/` ещё нет Parquet-файлов:

   ```bash
   python dataset/dataset_tool.py download
   python dataset/dataset_tool.py check
   ```

3. Загрузить модель эмбеддингов, если её ещё нет в кеше. Нужен доступ к сети:

   ```bash
   EMBEDDING_ALLOW_DOWNLOAD=true python -c 'from rag.embeddings.st import SentenceTransformerEmbeddings; SentenceTransformerEmbeddings(model_name="intfloat/multilingual-e5-small").encode_queries(["проверка"])'
   ```

   По умолчанию модель кешируется в `data/cache/embeddings/`. Другой путь можно
   задать через `SENTENCE_TRANSFORMERS_HOME`. После загрузки сеть для поиска не нужна.

## Запуск

Из корня проекта с активированным окружением:

```bash
python -m rag.rag_pipeline
```

Предварительно запускать ноутбук, сервер или отдельный сборщик индексов не нужно.
Скрипт сам читает первые 100 непустых документов, разбивает текст на чанки по
1024 символа с перекрытием 64, строит три индекса и выводит top-5 Hybrid
в виде контекста. Индексы хранятся в памяти и пересоздаются при каждом запуске.

Запрос меняется в `QUERY` внутри `rag_pipeline.py`; число документов, размеры
чанков и `k` — в `main()`. Эти параметры сейчас заданы в коде.
Скрипт не читает `.env` автоматически: переменные окружения нужно экспортировать
в терминале.

## Подключение LLM

В `rag_pipeline.py` после поиска уже есть `results` и строка `context`.
В этой точке нужно передать вопрос и контекст в модель, затем вернуть её ответ.
У каждого результата доступны `doc_id`, `chunk_id`, `text`, `score`, `metadata`;
в метаданных — заголовок, тип и дата документа.

При подключении к серверу загрузку документов и построение индексов стоит
выполнить один раз при старте. На каждый вопрос достаточно вызывать
`retrievers["hybrid"].retrieve(query, k=5)` и собирать контекст из результата.
