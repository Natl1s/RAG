# Backend — FastAPI-заглушка чат-бота

Минимальный API для «Юридического ассистента ПСБ». RAG и LLM заменены одной
заглушкой: на любой вопрос возвращается одна и та же строка — `"BLOB"`.

## Файлы

- `main.py` — приложение, эндпоинты и `generate_answer()`;
- `test_main.py` — pytest-тесты;
- `requirements.txt` / `requirements-dev.txt` — зависимости и тесты.

## Запуск

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Документация — http://127.0.0.1:8000/docs.

## API

| Метод | Путь | Назначение |
| --- | --- | --- |
| `GET` | `/` | описание сервиса |
| `GET` | `/api/health` | `{"status": "ok"}` |
| `POST` | `/api/chat` | поток SSE: `token*` → `done` |

`POST /api/chat` принимает `{"query": "ваш вопрос"}` и отдаёт
`text/event-stream`:

```
data: {"type": "token", "text": "BLOB"}

data: {"type": "done", "answer": "BLOB", "latencyMs": 694}

data: [DONE]
```

## Тесты

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Docker

```bash
docker build -t psb-rag-backend .
```

## Куда подключать реальные модели

Сейчас `generate_answer(query)` возвращает константу. Чтобы подключить
реальный конвейер, замените только эту функцию:

- **RAG** — поиск по `dataset/` и `rag/`, вернуть найденные источники;
- **LLM** — вызов модели (OpenAI/GigaChat/внутренняя LLM ПСБ) с промптом.

Формат событий `token` / `done` и фронтенд при этом менять не нужно.
