# RAG: подготовка документов и поиск контекста

Модуль `rag/` читает документы, очищает текст, разбивает его на чанки и
строит поисковые индексы BM25, Dense и Hybrid. Основной интерфейс — класс
`RAG`, настройки собраны в `RAGConfig`. Генерация ответа через LLM в этот
модуль не входит: результат поиска можно передать модели как контекст.

## Структура

| Файл или папка | Назначение |
| --- | --- |
| `config.py` | `RAGConfig`: настройки чанкинга, эмбеддингов и поиска; создание эмбеддера |
| `rag_pipeline.py` | `RAG`, построение индексов и поиск; функции `initialaze_rag` и `search_chunks` |
| `chunker.py` | `Chunker`: чтение, очистка и разбиение документов с параметрами из конфига |
| `ingest/load_dataset.py` | Преобразование пар `(название, текст)` в `Document` |
| `ingest/preprocess.py` | Очистка HTML и нормализация пробелов |
| `ingest/chunking.py` | Разбиение на логические блоки и перекрывающиеся чанки |
| `ingest/schema.py` | Структуры `Document` и `Chunk` |
| `embeddings/embedders.py` | Локальные и удалённые эмбеддинги |
| `retrieve.py` | Сборка словаря поисковиков через `build_retrievers` |
| `index/` | Реализации BM25, Dense на FAISS, Hybrid и контракт результата поиска |
| `log_config.py` | Логирование через `LOG_LEVEL` и `LOG_FILE` |
| `RAG_RusLaw_Demo.ipynb` | Ноутбук с примерами; его импорты и вызовы нужно сверять с текущим API |

## Подготовка окружения

Выполняйте команды из корня репозитория в окружении Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy httpx rank-bm25 faiss-cpu sentence-transformers huggingface-hub
```

Для локальных эмбеддингов по умолчанию используется модель из
`models/multilingual-e5-small`. Она должна быть доступна локально.
Загрузка модели происходит при первом расчёте эмбеддингов, на CPU.

Можно указать идентификатор Hugging Face вместо локального пути. Если модели
нет в кеше, для загрузки нужно установить `EMBEDDING_ALLOW_DOWNLOAD=true`.
Кеш по умолчанию расположен в `data/cache/embeddings`; путь меняется через
`SENTENCE_TRANSFORMERS_HOME` или `TRANSFORMERS_CACHE`.

## Использование RAG

```python
from rag.config import RAGConfig
from rag.rag_pipeline import RAG

# Названия используются как doc_id и title: задавайте уникальные значения.
documents = [
    ("документ-1", "<p>Договор определяет права и обязанности сторон.</p>"),
    ("документ-2", "Стороны несут ответственность за нарушение обязательств."),
]

config = RAGConfig(
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="models/multilingual-e5-small",
    retrieval_type="hybrid",
    top_k=5,
)
rag = RAG(config)
rag.build(documents)

results = rag.search_chunks("Какова ответственность сторон?")
for doc_id, text in results:
    print(doc_id, text)
```

`build()` принимает коллекцию пар `(название, текст)`. Несмотря на имя
аргумента `documents_path` и аннотацию `str` в текущем коде, путь к файлу или
папке передавать нельзя: чтение файлов, Parquet и JSONL здесь не реализовано.
Преобразуйте исходные данные в пары до вызова `build()`.

Внутри `build()` выполняются:

1. Чтение и очистка документов через `self.chunker.read_docs()`.
2. Разбиение через `self.chunker.chunk()`.
3. Построение поисковиков через `build_retrievers()`.

Индексы остаются в памяти в `rag.retrievers`. Сохранения на диск нет;
повторный вызов `build()` строит индексы заново. Вызывайте его один раз
для корпуса, затем используйте `search_chunks()` для каждого вопроса.

`search_chunks()` возвращает `list[tuple[str, str]]`: идентификатор документа
и текст фрагмента. Один документ может дать несколько результатов.
Для доступа к `chunk_id`, `score` и `metadata` используйте поисковик напрямую:

```python
results = rag.retrievers[config.retrieval_type].retrieve("Ответственность сторон", k=5)
for result in results:
    print(result.doc_id, result.chunk_id, result.score, result.text)
```

В модуле также есть короткие обёртки:

```python
from rag.rag_pipeline import initialaze_rag, search_chunks

rag = initialaze_rag(documents)  # Создаёт RAGConfig с настройками по умолчанию.
results = search_chunks("Ответственность сторон", rag)
```

Имя `initialaze_rag` приведено в точности как в коде.
Команда `python -m rag.rag_pipeline` сама не строит индексы и не выполняет
поиск: активной точки входа с примером запуска в модуле нет.

## Настройки RAGConfig

### Чанкер

| Поле | По умолчанию | Назначение |
| --- | --- | --- |
| `chunk_size` | `1000` | Максимальный размер чанка в символах, должен быть больше нуля |
| `chunk_overlap` | `200` | Перекрытие внутри логического блока: `0 <= overlap < chunk_size` |

Текст очищается от HTML-тегов, таблиц, изображений и хвостов `?pict...`;
пробелы и переносы нормализуются. Затем текст делится на логические блоки
по переносам, нумерованным пунктам и заголовкам «Статья», «Глава», «Раздел».
Перекрытие применяется внутри каждого блока. Пустые тексты пропускаются.

`Chunker(config)` можно использовать отдельно: `read_docs()` создаёт и очищает
документы, `clean_docs()` возвращает очищенные копии, `chunk()` возвращает
список `Chunk`. При разбиении текст также проходит очистку.

### Эмбеддер

| Поле | По умолчанию | Назначение |
| --- | --- | --- |
| `embedding_model` | `models/multilingual-e5-small` | Путь или идентификатор локальной модели |
| `embedding_type` | `local` | `local` — Sentence Transformers, `remote` — API |
| `batch_size` | `32` | Размер батча локального эмбеддера |
| `client` | `None` | `httpx.Client`, обязателен для `remote` |

`RAGConfig.__post_init__()` создаёт `config.embedder`. Локальный эмбеддер
добавляет префиксы `passage:` и `query:` и нормализует векторы.
Удалённый вызывает `script.embeddings.embeddings()` через `POST /embeddings`.

Для удалённого режима задайте переменные окружения **до импорта** модулей:

```bash
export LLM_BASE_URL="https://your-api.example/v1"
export EMBEDDER_API_KEY="<ключ>"
export EMB_MODEL="baai/bge-m3"
```

```python
import httpx
from rag.config import RAGConfig
from rag.rag_pipeline import RAG

with httpx.Client(timeout=120) as client:
    config = RAGConfig(embedding_type="remote", client=client)
    rag = RAG(config)
    rag.build(documents)
    results = rag.search_chunks("Ответственность сторон")
```

Клиент должен оставаться открытым при построении индекса и поиске.
Удалённая модель задаётся через `EMB_MODEL`, а не `embedding_model`.
Файл `.env` эти модули автоматически не загружают.

### Поиск и индексация

| Поле | По умолчанию | Фактическое поведение |
| --- | --- | --- |
| `retrieval_type` | `hybrid` | Выбирает поисковик: `bm25`, `dense` или `hybrid` |
| `top_k` | `5` | Количество результатов в `RAG.search_chunks()` |
| `use_dense` | `True` | Включает построение Dense-индекса FAISS |
| `use_hybrid` | `True` | Включает Hybrid при включённом Dense |
| `min_chunk_chars` | `50` | Минимальная длина чанка для Dense-индекса |
| `use_reranker` | `False` | Объявлено, но пока не используется |
| `use_faiss` | `True` | Объявлено, но пока не используется; Dense всегда использует FAISS |
| `max_chunk_chars` | `10000` | Объявлено, но пока не используется |

BM25 строится всегда. Dense использует `FAISS IndexFlatIP` с нормализованными
векторами. Hybrid объединяет выдачи BM25 и Dense с равными весами `0.5`,
нормализуя оценки по рангу и запрашивая у каждого поисковика `2 * top_k` кандидатов.

Значение `retrieval_type` должно соответствовать построенному поисковику.
Например, для поиска только по BM25:

```python
config = RAGConfig(retrieval_type="bm25", use_dense=False, use_hybrid=False)
```

BM25 не требует расчёта эмбеддингов. Пустой корпус нельзя индексировать;
Dense также выдаёт ошибку, если после фильтрации по `min_chunk_chars`
не осталось чанков. Размерность Dense-эмбеддингов должна быть не меньше 32.

В текущем `RAG` поле `self.embedder` содержит строку `config.embedding_model`.
При этом Dense-поиск рассчитывает векторы через объект `config.embedder`
в `build_retrievers()`.

## Передача контекста в LLM

```python
chunks = rag.search_chunks("Ответственность сторон")
context = "\n\n".join(f"Источник: {doc_id}\n{text}" for doc_id, text in chunks)
```

Передайте `context` вместе с вопросом в выбранную LLM. Сам `RAG` возвращает
найденные фрагменты и не вызывает генеративную модель.
