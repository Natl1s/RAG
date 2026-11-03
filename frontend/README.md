# Frontend — чат-бот «Юридический ассистент ПСБ»

Браузерный чат-бот на React + TypeScript + Vite с заглушками RAG/LLM.
Этот каталог можно запускать отдельно от бэкенда.

```bash
pnpm install
pnpm dev
```

Откройте http://localhost:5173.

- Режим **«Заглушка»** — всё работает в браузере без сервера.
- Режим **«API»** — нужен FastAPI-бэкенд из `../backend` (см. корневой README).

Основные файлы:

- `src/api/client.ts` — единая точка входа в модель (`mock` | `api`);
- `src/lib/mockEngine.ts` — браузерная заглушка RAG + LLM;
- `src/hooks/useChat.ts` — состояние чата и потоковый вывод;
- `src/components/` — интерфейс чата;
- `knowledge/knowledge.json` — база знаний (её же читает FastAPI-бэкенд).
