"""Минимальный пример сабмита ПСБ: embeddings + rerank + LLM.

Это учебный каркас, а не полноценное RAG-решение: используется только
простой поиск по словам среди документов и короткие текстовые фрагменты.
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET
from pathlib import Path
from rag.rag_pipeline import initialaze_rag, search_chunks
# import httpx


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(env(name))))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return max(0.0, min(2.0, float(env(name))))
    except ValueError:
        return default


# Настройки приходят с платформы. В ZIP НЕ НУЖНО класть секреты и .env.
DATA_DIR = Path(env("DATA_DIR", "data"))
# DATA_DIR = Path(__file__).resolve().parent / "dataset"
# API_BASE = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")).rstrip("/")
# LLM_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY")
# EMBED_KEY = env("EMBEDDER_API_KEY") or LLM_KEY
# RERANK_KEY = env("RERANKER_API_KEY") or LLM_KEY

# LLM_MODEL = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "deepseek/deepseek-v4.1-flash")
# EMB_MODEL = env("EMB_MODEL", "baai/bge-m3")
# RERANKER_MODEL = env("RERANKER_MODEL", "cohere/rerank-5-fast")

# TOP_K = env_int("TOP_K", 4, 1, 8)
# RERANKER_TOP_N = env_int("RERANKER_TOP_N", 2, 1, 5)
# MAX_TOKENS = env_int("MAX_TOKENS", 1024, 128, 8192)
MAX_WORKERS = env_int("MAX_WORKERS", 2, 1, 8)
# TEMPERATURE = env_float("TEMPERATURE", 0.0)
# CHUNKER = env("CHUNKER", "common")

# SNIPPET_CHARS = 1200
FALLBACK = "Не удалось получить ответ."
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{4,}")


def words(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower().replace("ё", "е")))


def find_file(*names: str) -> Path:
    print(DATA_DIR)
    for root in (DATA_DIR, Path(__file__).resolve().parent / DATA_DIR, Path.cwd()):
        for name in names:
            path = root / name
            if path.is_file():
                return path
    raise FileNotFoundError(f"Не найден один из файлов: {', '.join(names)}")


def load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("В questions.csv нужны колонки id и question")
        rows = [(str(r["id"]).strip(), str(r["question"]).strip()) for r in reader]
    if not rows or any(not qid for qid, _ in rows):
        raise ValueError("В questions.csv отсутствуют вопросы или id")
    return rows


def read_docx(raw: bytes) -> str:
    # DOCX — это ZIP с XML. Пример обходится без дополнительной библиотеки.
    with zipfile.ZipFile(io.BytesIO(raw)) as doc:
        xml = doc.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "\n".join(
        "".join(x.text or "" for x in paragraph.iter(ns + "t"))
        for paragraph in root.iter(ns + "p")
    )


def load_documents(path: Path) -> list[tuple[str, str]]:
    documents = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/") or name.startswith("__MACOSX/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in {".txt", ".md", ".docx"}:
                continue
            raw = archive.read(name)
            try:
                text = read_docx(raw) if suffix == ".docx" else raw.decode("utf-8-sig")
            except (UnicodeDecodeError, KeyError, ValueError, zipfile.BadZipFile, ET.ParseError):
                print(f"Пропущен нечитаемый документ: {name}", file=sys.stderr)
                continue
            if text.strip():
                documents.append((name, text.strip()))
    if not documents:
        raise ValueError("В архиве базы знаний нет читаемых документов")
    return documents


# def snippet(text: str) -> str:
#     # Минимальный пример применения CHUNKER, не полноценное разбиение на чанки.
#     piece = text[:SNIPPET_CHARS]
#     if CHUNKER == "common" and len(text) > SNIPPET_CHARS:
#         boundary = piece.rfind("\n")
#         if boundary > SNIPPET_CHARS // 2:
#             piece = piece[:boundary]
#     return piece.strip()


# def post(client: httpx.Client, path: str, key: str, payload: dict) -> dict:
#     if not key:
#         raise ValueError("На платформе отсутствует API-ключ")
#     response = client.post(
#         API_BASE + path,
#         headers={"Authorization": "Bearer " + key},
#         json=payload,
#     )
#     response.raise_for_status()
#     return response.json()


# def embeddings(client: httpx.Client, texts: list[str]) -> list[list[float]]:
#     """Пример вызова EMB_MODEL через POST /embeddings."""
#     result = post(client, "/embeddings", EMBED_KEY, {
#         "model": EMB_MODEL,
#         "input": texts,
#         "encoding_format": "float",
#     })
#     vectors: list[list[float] | None] = [None] * len(texts)
#     for position, item in enumerate(result["data"]):
#         index = int(item.get("index", position))
#         if index < 0 or index >= len(texts):
#             raise ValueError("Некорректный индекс embedding")
#         vector = item["embedding"]
#         norm = math.sqrt(sum(float(x) ** 2 for x in vector)) or 1.0
#         vectors[index] = [float(x) / norm for x in vector]
#     if any(v is None for v in vectors):
#         raise ValueError("Получены не все embeddings")
#     return vectors  # type: ignore[return-value]

def select_documents(question: str,
                     retrievers) -> list[tuple[str, str]]:
    
    return search_chunks(question, retrievers)


def generate(docs: list[tuple[str, str]]) -> str:
    return docs
    


def main() -> None:
    questions = load_questions(find_file("questions.csv"))
    documents = load_documents(find_file("kb.zip", "knowledge-base.zip", "knowledge_base.zip"))

    retrievers = initialaze_rag(documents)
    def work(position: int, question: str) -> tuple[int, str]:
        try:
            selected = select_documents(question, retrievers)
            return position, generate(selected)
        except Exception as exc:
            print(f"Ошибка вопроса {position}: {type(exc).__name__}", file=sys.stderr)
            return position, FALLBACK

    
    answers = [FALLBACK] * len(questions)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(work, i, question) for i, (_, question) in enumerate(questions)]
        for finished, future in enumerate(as_completed(futures), 1):
            index, answer = future.result()
            answers[index] = answer
            if finished % 10 == 0 or finished == len(questions):
                print(f"Обработано {finished}/{len(questions)}", flush=True)

    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for i, (qid, _) in enumerate(questions):
            writer.writerow([qid, answers[i]])
    print(f"Создан submission.csv: {len(questions)} ответов", flush=True)


if __name__ == "__main__":
    main()
