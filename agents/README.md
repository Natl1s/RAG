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
4. Если ключ не задан, приложение работает на RAG-fallback и пишет
   предупреждение в консоль.

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

### Тесты

```bash
.venv/bin/python -m unittest discover -s agents/tests -v
```
