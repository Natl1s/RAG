# Frontend — чат-бот «Юридический ассистент ПСБ»

Браузерный чат-бот на React + TypeScript + Vite. Ответы приходят потоком SSE
от FastAPI-бэкенда из `../backend`.

```bash
npm install
npm run dev
```

Откройте http://localhost:5173 — нужен запущенный бэкенд (`uvicorn main:app --port 8000`).

Основные файлы:

- `src/api/apiSse.ts` — `POST /api/chat`, чтение SSE, история диалога;
- `src/conversation.ts` — `conversation_id` вкладки в `sessionStorage`,
  восстановление после F5, beacon при закрытии;
- `src/hooks/useChat.ts` — состояние чата и потоковый вывод;
- `src/components/` — интерфейс чата.

В Docker фронтенд собирается в `frontend/Dockerfile` и раздаётся nginx
(`nginx.conf`): `/api/` проксируется на контейнер `backend`, буферизация SSE
отключена.
