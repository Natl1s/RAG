# Как запускать

## Настройка LLM (локально)

1. Скопируйте шаблон окружения и ограничьте права на файл:
   ```bash
   cp .env.example .env
   chmod 600 .env
   ```
2. Впишите свой `LLM_API_KEY` в `.env`. Ключ SiliconFlow можно получить здесь:
   https://cloud.siliconflow.com/me/account/ak
3. Файл `.env` не коммитится и не попадает в Docker-образ. У каждого
   разработчика свой локальный `.env`.
4. Настройки LLM обязательны: без `LLM_BASE_URL` / `LLM_API_KEY` /
   `LLM_MODEL` агент падает с понятной ошибкой при старте.

Проверить настройки без вывода ключа:
```bash
.venv/bin/python -c "from agents.config import get_settings; s=get_settings(); print('base_url:', s.llm_base_url, 'model:', s.llm_model, 'key_set:', bool(s.llm_api_key))"
```

## Использование

RAG-индекс строится снаружи до запуска агента. Агент не создаёт индекс и не
импортирует внутренние модули RAG: он получает готовые `retrievers` и
обращается к `rag.rag_pipeline.search_chunks`.

```python
from agents.bootstrap import build_local_service

# retrievers построены внешним кодом/сервисом
service = build_local_service(retrievers)
print(service.answer("u-1", "Какие налоги платит ИП?"))
```

## Логи

Локальный запуск через `build_local_service` включает логирование: консоль +
файл `logs/llm_<дата_время запуска>.log` в корне репозитория. Один файл на
запуск: время запуска — в имени файла и в первой строке. В лог попадают
промпты и сырые ответы LLM, поэтому после запуска можно изучить, какие
запросы уходили в модель и что она ответила.

Переменные окружения:

- `LOG_LEVEL` (по умолчанию `INFO`) — порог уровня логов. `WARNING` прячет
  промпты и ответы, `DEBUG` показывает больше деталей.
- `LLM_LOG_FILE` — путь к файлу логов. `LLM_LOG_FILE=off` отключает файл
  (останется только консоль). По умолчанию на каждый запуск создаётся новый
  файл в `logs/`.

Для итогового решения логирование в файл не включается: `setup_logging`
вызывается только внутри `build_local_service`, который backend и скрипты
сабмита не используют. Логи не коммитятся: `*.log` уже в `.gitignore`.

### Тесты

```bash
.venv/bin/python -m unittest discover -s agents/tests -v
```
