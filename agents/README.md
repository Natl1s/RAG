# Как запускать

в командной строке
```bash
EMBEDDING_ALLOW_DOWNLOAD=true .venv/bin/python -m agents.run u-1 "Какие налоги платит ИП?"
```

Или из Python:
```python
from agents.service import AgentService
from agents.bootstrap import build_local_service

service = build_local_service(limit=100)
print(service.answer("u-1", "Какие налоги платит ИП?"))
```

### Тесты:
```
.venv/bin/python -m unittest discover -s agents/tests -v
```
Интеграционный тест:

```
RUN_RAG_INTEGRATION=1 EMBEDDING_ALLOW_DOWNLOAD=true \
  .venv/bin/python -m unittest agents.tests.test_integration -v
```