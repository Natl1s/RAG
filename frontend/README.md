# Frontend — чат-бот «Юридический ассистент ПСБ»

Браузерный чат-бот на React + TypeScript + Vite. Ответы приходят потоком SSE
от FastAPI-бэкенда из `../backend`.

```bash
pnpm install
pnpm dev
```

Откройте http://localhost:5173 — нужен запущенный бэкенд (`uvicorn main:app --port 8000`).

Основные файлы:

- `src/api/apiSse.ts` — чтение потока SSE;
- `src/hooks/useChat.ts` — состояние чата и потоковый вывод;
- `src/components/` — интерфейс чата.
