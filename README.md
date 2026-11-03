# Юридический ассистент ПСБ — чат-бот

Чат-бот в браузере: задаёте вопрос, ответ приходит потоком (SSE) и печатается
по токенам.

> Демо-режим: **RAG** и **LLM** заменены одной заглушкой, которая на любой
> вопрос возвращает одну и ту же строку. Место подключения реальных моделей —
> `generate_answer()` в `backend/main.py`; интерфейс `AskEvent` менять не нужно.

## Что умеет

- Чат в браузере с потоковым выводом ответа (эффект печати).
- Остановка генерации, копирование ответа, очистка чата, адаптивная вёрстка.
- Минимальный API: `GET /`, `GET /api/health`, `POST /api/chat` (SSE).

## Архитектура

```
psb-rag/
├── frontend/                    # React + TypeScript + Vite — браузерный чат-бот
│   ├── src/
│   │   ├── api/apiSse.ts        # чтение потока SSE
│   │   ├── hooks/useChat.ts     # состояние чата и потоковый вывод
│   │   └── components/…         # Header, Chat, MessageBubble, Composer
│   ├── Dockerfile               # сборка и раздача через nginx
│   └── nginx.conf               # прокси /api → backend, SSE без буферизации
├── backend/                     # FastAPI-заглушка
│   ├── main.py                  # /api/chat, /api/health, generate_answer()
│   ├── test_main.py             # pytest-тесты
│   └── Dockerfile
├── rag/                         # зона реального RAG (см. rag/README.md)
├── agents/                      # зона реальных LLM-агентов (см. agents/README.md)
├── dataset/                     # данные: raw / processed / index (не в образе)
├── docker-compose.yml
└── README.md
```

Бэкенд общается с UI через события `AskEvent`:

- `token` — ассистент печатает ответ по токенам;
- `done` — итоговый ответ и время генерации.

Когда появятся реальные модели, меняется только `generate_answer()` в
`backend/main.py` (и папки `rag/`, `agents/`), фронтенд не трогаем.

## Запуск бэкенда

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Проверка: `GET /api/health`, документация — http://127.0.0.1:8000/docs.

### API

| Метод | Путь | Назначение |
| --- | --- | --- |
| `GET` | `/` | описание сервиса |
| `GET` | `/api/health` | статус |
| `POST` | `/api/chat` | поток SSE: `token*` → `done` |

## Запуск фронтенда

Требуется Node.js 18+ и pnpm (или npm).

```bash
cd frontend
pnpm install     # или npm install
pnpm dev         # или npm run dev
```

Откройте http://localhost:5173. Vite проксирует `/api/*` на
`http://127.0.0.1:8000`, поэтому бэкенд должен быть запущен.

## Запуск в Docker

```bash
docker compose up --build -d
```

- чат: http://localhost:8080
- API: http://localhost:8000/api/health

Nginx раздаёт собранный фронтенд и проксирует `/api/` на контейнер `backend`.
Буферизация для SSE отключена, поэтому ответ приходит по токенам.

## Тесты бэкенда

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q
```

## Сборка для продакшена

```bash
cd frontend
pnpm build      # tsc --noEmit + vite build → dist/
pnpm preview    # локальный просмотр собранного бандла
```

## Стек

React · TypeScript · Vite · FastAPI · SSE · Docker · nginx
