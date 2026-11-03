# Временный датасет RusLawOD

Скрипт `dataset_tool.py` предназначен для загрузки и просмотра временных данных, используемых при разработке RAG-системы.
Источник данных: https://huggingface.co/datasets/irlspbru/RusLawOD
### Установить временные данные

```bash
python3 dataset/dataset_tool.py download
```

### Посмотреть пример документа (первые 1000 символов)

```bash
python3 dataset/dataset_tool.py example
```

### Проверить целостность данных

```bash
python3 dataset/dataset_tool.py check
```

Данные сохраняются локально и не загружаются повторно при обычной работе с ними.
