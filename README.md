# Юридический ассистент ПСБ — чат-бот

Браузерный чат по внутренней базе знаний: вопрос уходит в FastAPI, ответ
приходит потоком SSE и печатается по мере поступления. Диалог помнит контекст
внутри вкладки, переживает F5 и удаляется вместе с вкладкой.

## Быстрый старт

Нужен только Docker. Из корня репозитория:

```bash
docker compose up --build
```

- чат — http://localhost:8080
- API — http://localhost:8000/api/health
- документация — http://localhost:8000/docs

Сборка образов, установка зависимостей, скачивание корпуса и сборка фронтенда
происходят внутри этой команды. Значения `LLM_*` Compose берёт из `.env`
рядом с `docker-compose.yml`; сам файл в образ не копируется.

Проверка готовности:

```bash
curl http://localhost:8080/api/health
# {"status":"ok","agent":true,"retrieval":"hybrid","conversations":0}
```

`retrieval` показывает, что реально собралось на старте: `hybrid` (BM25 +
векторный поиск) или `bm25` (если embedding-эндпоинт недоступен — сервис
поднимается в деградированном режиме, а не падает).

## Что умеет

- потоковый ответ (`meta` → `token*` → `done`), остановка генерации, копирование;
- отдельный диалог на каждую вкладку: `conversation_id` в `sessionStorage`;
- восстановление истории после перезагрузки страницы;
- удаление диалога при закрытии вкладки (beacon + 15-секундная задержка);
- очистка диалога кнопкой «Очистить» (удаляет историю на сервере и выдаёт новый id);
- безопасная деградация: BM25 без эмбеддингов и автоматическое отключение
  внешнего реранкера при ошибке.

## Архитектура

```
psb-rag/
├── frontend/                 # React + TypeScript + Vite
│   ├── src/api/apiSse.ts     # POST /api/chat, разбор SSE
│   ├── src/conversation.ts   # id вкладки, restore, beacon при закрытии
│   ├── src/hooks/useChat.ts  # состояние чата
│   ├── Dockerfile            # сборка и раздача через nginx
│   └── nginx.conf            # прокси /api → backend без буферизации
├── backend/                  # FastAPI: HTTP-слой и память диалогов
│   ├── main.py               # /api/chat, /api/conversations/*, warm_up()
│   ├── test_main.py          # pytest: поток, история, деградация
│   └── Dockerfile
├── rag/                      # поиск: BM25, FAISS, гибрид, реранкер
├── agents/                   # LangGraph-агент: разбор диалога, генерация
├── script/embeddings.py      # вызовы /embeddings и /rerank
├── dataset/result_chunks.csv # готовый корпус чанков (680 строк, 21 документ)
├── docker-compose.yml
└── .env.example              # только имена переменных, без значений
```

Поток запроса:

1. фронтенд отправляет только новое сообщение и `conversation_id`;
2. бэкенд достаёт историю из памяти процесса, обрезает её (8 реплик / 6000
   символов) и склеивает в строку с метками `Пользователь:` / `Ассистент:`;
3. агент детерминированно разбирает строку (`agents/dialog.py`), ищет чанки
   в RAG и генерирует ответ;
4. бэкенд сохраняет ответ в историю и отдаёт его потоком SSE.

## API

| Метод | Путь | Назначение |
| --- | --- | --- |
| `GET` | `/` | описание сервиса |
| `GET` | `/api/health` | статус, режим поиска, число диалогов |
| `POST` | `/api/chat` | поток SSE: `meta` → `token*` → `done` |
| `GET` | `/api/conversations/{id}` | история диалога |
| `DELETE` | `/api/conversations/{id}` | удалить диалог |
| `POST` | `/api/conversations/{id}/close` | сигнал «вкладка закрылась» |

```bash
curl -N http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Какая арендная плата по договору аренды?","conversation_id":"tab-1"}'
```

Подробнее о событиях и жизненном цикле диалога — `backend/README.md`.

## Переменные окружения

Шаблон — `.env.example` (только имена). Локальный `.env` в корне:

```env
LLM_BASE_URL=https://api.siliconflow.com/v1
LLM_API_KEY=...
LLM_MODEL=deepseek-ai/DeepSeek-V3
EMB_MODEL=Qwen/Qwen3-Embedding-0.6B
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
```

- `LLM_*` (или `OPENAI_*`) — чат-модель;
- `EMB_MODEL` — модель эмбеддингов; при ошибке сервис переходит на BM25;
- `RERANKER_MODEL` — модель реранкера; при ошибке отключается на лету;
- остальные лимиты (`TOP_K`, `HISTORY_LIMIT`, `MAX_*`) необязательны, значения
  по умолчанию заданы в коде.

`.env` не коммитится и не попадает в образ. На стенде значения подставляет
платформа; переменные окружения имеют приоритет над `.env`.

## Локальная разработка

Бэкенд:

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows, или source .venv/bin/activate
pip install -r backend/requirements-dev.txt
cd backend
uvicorn main:app --reload --port 8000
```

Фронтенд (Node.js 18+):

```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173, /api проксируется на :8000
```

## Тесты

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q
```

Тесты не поднимают реальный RAG: агент подменяется заглушкой, проверяются
поток SSE, накопление и изоляция истории, восстановление после перезагрузки,
закрытие вкладки, вырезание `Источники:` и деградация поиска.

## Docker: детали

```bash
docker compose up --build        # собрать и запустить
docker compose logs -f backend   # логи RAG и агента
docker compose down              # остановить
```

Тяжёлые загрузки по умолчанию выключены. Скачать исходный датасет RusLawOD
в том (нужен только для пересборки корпуса):

```bash
DOWNLOAD_DATASET=true docker compose up --build
```

## Стек

React · TypeScript · Vite · FastAPI · LangGraph · BM25 + FAISS · SSE · Docker · nginx
