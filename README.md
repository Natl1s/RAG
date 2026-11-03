# Юридический ассистент ПСБ — чат-бот с заглушками моделей

Интеллектуальный ассистент для работы с юридическими документами: ищет релевантную
информацию в базе знаний, отвечает на вопросы пользователя и корректно обрабатывает
вопросы, на которые в базе **нет** ответа (предлагает переформулировать вопрос,
показывает ближайшие темы и передаёт вопрос юристу).

> Демо-режим: модели **RAG** и **LLM** заменены заглушками. Места подключения
> реальных моделей изолированы в одном интерфейсе `AskEvent`, чтобы заменить
> заглушку на настоящий конвейер без изменения UI.

## Что умеет

- Чат в браузере с потоковым выводом ответа (эффект печати).
- Поиск по базе юридических знаний с показом источников, цитат и релевантности.
- Статусы `answered` / `not_found`: если ответа нет, ассистент честно сообщает об
  этом, показывает ближайшие темы и предлагает передать вопрос юристу.
- Эскалация к юристу с номером обращения (в демо — имитация, в API — `POST /api/escalate`).
- Два режима: браузерная заглушка (`mock`) и FastAPI-заглушка (`api`, SSE).
- Остановка генерации, копирование ответа, очистка чата, адаптивная вёрстка.

## Архитектура

```
psb-rag/
├── frontend/                    # React + TypeScript + Vite — браузерный чат-бот
│   ├── src/
│   │   ├── api/client.ts        # единая точка входа в «модель» (mock | api)
│   │   ├── api/apiSse.ts        # клиент FastAPI (Server-Sent Events)
│   │   ├── lib/mockEngine.ts    # браузерная заглушка RAG + LLM
│   │   ├── data/…               # загрузка базы знаний
│   │   ├── hooks/useChat.ts     # состояние чата и потоковый вывод
│   │   └── components/…         # Header, Chat, MessageBubble, Sources, Composer
│   ├── knowledge/knowledge.json # база знаний (фронтенд и бэкенд)
│   ├── Dockerfile               # сборка и раздача через nginx
│   └── nginx.conf               # прокси /api → backend, SSE без буферизации
├── backend/                     # FastAPI-заглушка (SSE, та же логика)
│   ├── main.py                  # /api/chat, /api/escalate, /api/health
│   ├── test_main.py             # pytest-тесты
│   └── Dockerfile
├── rag/                         # реальный RAG: чанкинг, индекс, поиск
├── agents/                      # реальные LLM-агенты: промпты, вызов модели
├── dataset/                     # данные: raw / processed / index (не в образе)
├── docker-compose.yml
└── README.md
```

Папки `rag/`, `agents/` и `dataset/` — зоны ответственности команды моделей.
Они лежат на одном уровне с `backend/` и `frontend/` и пока содержат только
README с контрактами (`rag/README.md`, `agents/README.md`, `dataset/README.md`).
Бэкенд сейчас работает на заглушках; когда появятся реальные реализации,
меняется только вызов внутри `backend/main.py`, а события `retrieval` /
`token` / `done` и фронтенд остаются прежними.

Каталог `frontend/` самодостаточен: внутри лежат исходники, база знаний и конфиги,
поэтому его можно запускать и передавать отдельно от бэкенда.

Конвейер «RAG + LLM» общается с UI через единый набор событий `AskEvent`:

- `retrieval` — ретривер нашёл источники (чанки с релевантностью);
- `token`    — LLM стримит ответ по токенам;
- `done`     — итоговый ответ, источники, уверенность, статус и задержка.

Статусы ответа: `answered` (ответ найден в базе) и `not_found` (ответа нет).

## Быстрый старт (без бэкенда)

Требуется Node.js 18+ и pnpm (или npm).

```bash
cd frontend
pnpm install     # или npm install
pnpm dev         # или npm run dev
```

Откройте http://localhost:5173 — чат работает полностью в браузере на заглушках
(режим «Заглушка» в шапке). Пример вопроса без ответа в базе:
«Какие документы нужны для регистрации ООО?».

Режим по умолчанию можно задать через `frontend/.env` (см. `.env.example`):
`VITE_API_MODE=mock` или `VITE_API_MODE=api`.

## Запуск с FastAPI-заглушкой

В отдельном терминале:

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Затем переключите чат в режим **API** (переключатель в шапке). Vite проксирует
`/api/*` на `http://127.0.0.1:8000`. Проверить бэкенд: `GET /api/health`,
интерактивная документация — http://127.0.0.1:8000/docs.

### API

| Метод | Путь | Назначение |
| --- | --- | --- |
| `GET` | `/api/health` | статус и количество чанков в базе |
| `POST` | `/api/chat` | поток SSE: `retrieval` → `token*` → `done` |
| `POST` | `/api/escalate` | создать обращение к юристу (заглушка CRM) |

## Запуск в Docker

Один раз создайте отдельный билдер (см. раздел ниже, зачем он нужен):

```bash
docker buildx create --name psb-builder --driver docker-container --bootstrap
```

Дальше запуск:

```bash
docker compose build --builder psb-builder
docker compose up -d
```

Либо одной командой с переменной окружения:

```bash
BUILDX_BUILDER=psb-builder docker compose up --build -d
```

```powershell
$env:BUILDX_BUILDER = "psb-builder"; docker compose up --build -d
```

- чат: http://localhost:8080
- API: http://localhost:8000/api/health
- в шапке чата выберите режим **API**.

Nginx раздаёт собранный фронтенд и проксирует `/api/` на контейнер `backend`.
Буферизация для SSE отключена, поэтому ответ приходит по токенам.

### Если сборка фронтенда падает (`tsc: not found` / `/app/dist: not found`)

На некоторых установках Docker Desktop встроенный BuildKit некорректно запускает
процессы Node внутри `docker build`: `node -v` может выполняться ~30 секунд и
терять вывод, а `npm install` — завершаться мгновенно, ничего не устанавливая.
При этом `docker run node:20-alpine node -v` работает нормально.

Поэтому и нужен билдер с драйвером `docker-container` — в нём Node работает
штатно:

```bash
docker buildx create --name psb-builder --driver docker-container --bootstrap
docker compose build --builder psb-builder
docker compose up -d
```

Флаг `--builder` у `docker compose build` работает в любой консоли. Если задать
`BUILDX_BUILDER=psb-builder` в переменных среды Windows, то обычный
`docker compose up --build` тоже будет использовать нужный билдер (в уже
открытом терминале переменная не подхватится — откройте новый).

## Тесты бэкенда

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q
```

## Куда подключать реальные модели

Замените только реализацию источника событий, интерфейс `AskEvent` менять не нужно:

- **RAG**: `frontend/src/lib/mockEngine.ts` → `runRetrieval()` или
  `backend/main.py` → `run_retrieval()` → подключить векторную БД/эмбеддинги
  (например, pgvector, Qdrant, FAISS).
- **LLM**: та же функция `generateAnswer()` → вызов реальной модели
  (OpenAI/GigaChat/внутренняя LLM ПСБ) с промптом и извлечённым контекстом.
- **Эскалация**: `POST /api/escalate` в `backend/main.py` → создание заявки в
  CRM/тикет-системе (Jira, Bitrix24 и т.п.).

## Сборка для продакшена

```bash
cd frontend
pnpm build      # tsc --noEmit + vite build → dist/
pnpm preview    # локальный просмотр собранного бандла
```

## Стек

React · TypeScript · Vite · FastAPI · SSE · RAG/LLM (заглушки) · Docker · nginx
