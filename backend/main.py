"""FastAPI-бэкенд чат-бота «Юридический ассистент ПСБ».

Диалог хранится в памяти процесса. Фронт присылает только новое сообщение и
``conversation_id``; бэкенд рендерит всю историю строкой с метками
«Пользователь:» / «Ассистент:» и передаёт её агенту, который разбирает текст
детерминированно (``agents/dialog.py``). Ответ отдаётся потоком SSE:
``meta`` -> ``token*`` -> ``done`` -> ``[DONE]``.

Запуск из каталога ``backend``::

    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

# .env должен быть загружен ДО импорта rag: script.embeddings читает
# LLM_BASE_URL / LLM_API_KEY на уровне модуля, а не при вызове.
load_dotenv(_REPO_ROOT / ".env", override=False)

import httpx  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Response  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agents.bootstrap import build_local_service  # noqa: E402
from agents.service import AgentService  # noqa: E402
from rag.config import RAGConfig  # noqa: E402
from rag.rag_pipeline import RAG  # noqa: E402

DOCS_PATH = _REPO_ROOT / "dataset" / "result_chunks.csv"

MAX_TURNS = 8              # реплик в одном запросе к агенту (= history_limit)
MAX_CHARS = 6000           # суммарная длина рендера
MAX_CONVERSATIONS = 500    # сколько диалогов держим одновременно
IDLE_TTL = 30 * 60         # секунд без запросов до удаления
CLOSE_GRACE = 15           # секунд между /close и удалением
WARM_UP_ATTEMPTS = 3       # попытки собрать RAG на старте
WARM_UP_DELAY = 5          # пауза между попытками, секунд

ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")

ROLE = {"user": "Пользователь", "assistant": "Ассистент"}
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# Метка роли в начале строки. Отступ не спасает: в agents/dialog.py
# регэксп тоже начинается с ^\s*, поэтому двоеточие заменяем на U+2236.
_ROLE_LINE = re.compile(
    r"(?im)^(\s*)(Пользователь|Ассистент|Система|User|Assistant|System)(\s*):"
)

# Финальный блок ответа агента: «Источники: …».
_SOURCES_RE = re.compile(r"(?im)^\s*Источники\s*:")


@dataclass
class Conversation:
    """Один диалог: реплики плюс служебные метки для уборки."""

    turns: list[dict[str, str]] = field(default_factory=list)
    closing_at: float | None = None
    touched_at: float = field(default_factory=time.monotonic)
    last_seen_ms: int = 0


AGENT: AgentService | None = None
_HISTORY: dict[str, Conversation] = {}
_CHAT_LOCKS: dict[str, asyncio.Lock] = {}
_CLIENT: httpx.Client | None = None

logger = logging.getLogger("backend")

RETRIEVAL_MODE = "hybrid"


# ---------------------------------------------------------------------------
# Подготовка корпуса и агента
# ---------------------------------------------------------------------------


def _build_rag(client: httpx.Client) -> RAG:
    """Собрать индекс: гибридный, а при отказе эмбеддингов — только BM25.

    BM25 не требует сети, поэтому контейнер поднимется даже при недоступном
    или неверно названном embedding-эндпоинте. Качество поиска ниже, но чат
    работает, а причина видна в логе.
    """
    global RETRIEVAL_MODE
    documents = pd.read_csv(DOCS_PATH)

    rag_client = RAG(
        RAGConfig(
            embedding_type="remote",
            client=client,
            use_reranker=True,
        )
    )
    try:
        # Пробный вызов: одна ошибка вместо 22 неудачных батчей при сборке.
        rag_client.embedder.encode_queries(["проверка доступности"])
        rag_client.build(documents)
    except Exception as exc:  # noqa: BLE001 — причину пишем в лог
        logger.error(
            "Гибридный индекс не собрался (%s: %s). Перехожу на BM25.",
            type(exc).__name__,
            exc,
        )
    else:
        RETRIEVAL_MODE = "hybrid"
        return rag_client

    fallback = RAG(
        RAGConfig(
            embedding_type="remote",
            client=client,
            retrieval_type="bm25",
            use_dense=False,
            use_faiss=False,
            use_hybrid=False,
            use_reranker=False,
        )
    )
    fallback.build(documents)
    RETRIEVAL_MODE = "bm25"
    return fallback


class ResilientRAG:
    """Прокси к RAG: чинит поиск, не меняя ``rag/``.

    Две деградации по цепочке:

    1. ошибка реранкера — выключаем его и повторяем поиск;
    2. поиск всё ещё падает (например, умер embedding-эндпоинт) — переключаем
       ``retrieval_type`` на BM25, который не требует сети.
    """

    def __init__(self, rag_client: RAG) -> None:
        self._rag = rag_client

    def search_chunks(self, query: str):
        try:
            return self._rag.search_chunks(query)
        except Exception as exc:  # noqa: BLE001 — внешние API падают
            if not self._rag.config.use_reranker:
                return self._fallback_to_bm25(query, exc)
            logger.warning(
                "Реранкер недоступен (%s: %s). Отключаю и повторяю поиск.",
                type(exc).__name__,
                exc,
            )
            self._rag.config.use_reranker = False
            try:
                return self._rag.search_chunks(query)
            except Exception as retry_exc:  # noqa: BLE001
                return self._fallback_to_bm25(query, retry_exc)

    def _fallback_to_bm25(self, query: str, exc: Exception):
        config = self._rag.config
        if config.retrieval_type == "bm25" or "bm25" not in self._rag.retrievers:
            raise exc
        logger.warning(
            "Поиск «%s» недоступен (%s: %s). Перехожу на BM25.",
            config.retrieval_type,
            type(exc).__name__,
            exc,
        )
        config.retrieval_type = "bm25"
        config.use_reranker = False
        return self._rag.search_chunks(query)


def warm_up() -> None:
    """Прочитать чанки, собрать RAG-индексы и создать агента.

    Выполняется синхронно на старте приложения: до первого запроса сервер
    всё равно не отвечает, а блокировка event loop здесь безопасна.
    """
    global AGENT, _CLIENT
    if AGENT is not None:
        return
    if not DOCS_PATH.is_file():
        raise RuntimeError(f"Не найден корпус чанков: {DOCS_PATH}")

    _CLIENT = httpx.Client(timeout=90.0)
    rag_client = ResilientRAG(_build_rag(_CLIENT))
    AGENT = build_local_service(rag_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _CLIENT
    last_error: Exception | None = None
    for attempt in range(1, WARM_UP_ATTEMPTS + 1):
        try:
            warm_up()
            break
        except Exception as exc:  # noqa: BLE001 — показываем причину в логе
            last_error = exc
            logger.warning(
                "Сборка RAG не удалась (попытка %s/%s): %s",
                attempt,
                WARM_UP_ATTEMPTS,
                exc,
            )
            if _CLIENT is not None:
                _CLIENT.close()
                _CLIENT = None
            await asyncio.sleep(WARM_UP_DELAY)
    else:
        raise RuntimeError(f"Не удалось собрать RAG: {last_error}") from last_error

    task = asyncio.create_task(sweeper())
    try:
        yield
    finally:
        task.cancel()
        if _CLIENT is not None:
            _CLIENT.close()


app = FastAPI(title="PSB Legal Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Диалог: рендер, обрезка, хранение
# ---------------------------------------------------------------------------


def neutralize(text: str) -> str:
    """Не дать пользователю подделать метку роли новой строкой."""
    return _ROLE_LINE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{match.group(3)}\u2236",
        text,
    )


def _for_prompt(turn: dict[str, str]) -> str:
    """Ответ ассистента без финального блока «Источники:».

    Полный текст остаётся в истории для восстановления после F5, но в
    промпт список источников не нужен: агент всё равно ищет заново.
    """
    content = turn["content"]
    if turn["role"] != "assistant":
        return content
    match = _SOURCES_RE.search(content)
    return content[: match.start()].rstrip() if match else content


def render_dialog(turns: list[dict[str, str]]) -> str:
    """Склеить реплики в строку, которую понимает ``parse_dialog``."""
    return "\n".join(
        f"{ROLE[turn['role']]}: {neutralize(_for_prompt(turn))}"
        for turn in turns
        if turn["content"].strip()
    )


def trim(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """Оставить последние реплики в пределах лимитов агента."""
    turns = list(turns)[-MAX_TURNS:]
    total = sum(len(turn["content"]) for turn in turns)
    while len(turns) > 1 and total > MAX_CHARS:
        total -= len(turns.pop(0)["content"])
    # История не должна начинаться ответом ассистента без вопроса.
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    # Последняя реплика — всегда вопрос пользователя.
    while turns and turns[-1]["role"] != "user":
        turns.pop()
    return turns


def _conversation(cid: str) -> Conversation:
    conversation = _HISTORY.get(cid)
    if conversation is None:
        if len(_HISTORY) >= MAX_CONVERSATIONS:
            oldest = min(_HISTORY, key=lambda key: _HISTORY[key].touched_at)
            _HISTORY.pop(oldest, None)
            _CHAT_LOCKS.pop(oldest, None)
        conversation = Conversation()
        _HISTORY[cid] = conversation
    return conversation


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _touch(conversation: Conversation, client_ts: int | None) -> None:
    """Отметить активность и отменить отложенное удаление.

    ``client_ts`` — время браузера в миллисекундах. Оно нужно, чтобы
    отличить перезагрузку от закрытия: beacon может прийти уже после
    запроса новой страницы, но у него будет меньшая метка времени.
    """
    stamp = client_ts if client_ts and client_ts > 0 else _now_ms()
    conversation.last_seen_ms = max(conversation.last_seen_ms, stamp)
    conversation.closing_at = None
    conversation.touched_at = time.monotonic()


def _expired(now: float) -> list[str]:
    """Диалоги, которые пора удалить (протухшие или закрытые)."""
    return [
        cid
        for cid, conversation in _HISTORY.items()
        if now - conversation.touched_at > IDLE_TTL
        or (
            conversation.closing_at is not None
            and now - conversation.closing_at > CLOSE_GRACE
        )
    ]


async def sweeper() -> None:
    """Периодически чистить историю: TTL простоя и закрытые вкладки."""
    while True:
        await asyncio.sleep(5)
        for cid in _expired(time.monotonic()):
            _HISTORY.pop(cid, None)
            _CHAT_LOCKS.pop(cid, None)


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def chunk_text(text: str, size: int = 40) -> list[str]:
    """Нарезать готовый ответ на куски по словам — для эффекта печати."""
    chunks: list[str] = []
    buffer = ""
    for token in re.findall(r"\S+\s*", text):
        buffer += token
        if len(buffer) >= size:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


# ---------------------------------------------------------------------------
# Схемы и эндпоинты
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    ts: int | None = None


async def _stream(cid: str, query: str, ts: int | None = None) -> AsyncGenerator[str, None]:
    started = time.monotonic()
    # meta уходит сразу: браузер получает заголовки и не считает запрос мёртвым.
    yield sse({"type": "meta", "conversationId": cid})

    lock = _CHAT_LOCKS.setdefault(cid, asyncio.Lock())
    async with lock:
        conversation = _conversation(cid)
        _touch(conversation, ts)

        user_turn = {"role": "user", "content": query}
        # Повторная отправка того же вопроса (ретрай после ошибки) не дублируется.
        if not conversation.turns or conversation.turns[-1] != user_turn:
            conversation.turns.append(user_turn)

        rendered = render_dialog(trim(conversation.turns))

        try:
            if AGENT is None:
                raise RuntimeError("Агент не инициализирован")
            result = await AGENT.aanswer(cid, rendered)
            answer = (result.answer or "").strip() or "Не удалось сформировать ответ."
        except Exception as exc:  # noqa: BLE001 — отдаём текст ошибки во фронт
            yield sse({"type": "error", "message": str(exc)})
            yield sse(
                {"type": "done", "answer": "", "latencyMs": _elapsed_ms(started)}
            )
            yield "data: [DONE]\n\n"
            return

        conversation.turns.append({"role": "assistant", "content": answer})
        conversation.touched_at = time.monotonic()

        for piece in chunk_text(answer):
            yield sse({"type": "token", "text": piece})
            await asyncio.sleep(0)

        yield sse(
            {"type": "done", "answer": answer, "latencyMs": _elapsed_ms(started)}
        )
        yield "data: [DONE]\n\n"


@app.get("/")
def root() -> dict:
    return {
        "service": "PSB Legal Assistant API",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "agent": AGENT is not None,
        "retrieval": RETRIEVAL_MODE,
        "conversations": len(_HISTORY),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Пустой вопрос")

    cid = (req.conversation_id or "").strip() or uuid.uuid4().hex
    if not ID_RE.match(cid):
        raise HTTPException(status_code=422, detail="Некорректный conversation_id")

    return StreamingResponse(
        _stream(cid, query, req.ts),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.get("/api/conversations/{cid}")
def get_conversation(cid: str, ts: int | None = None) -> dict:
    conversation = _HISTORY.get(cid)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    # Запрос после перезагрузки страницы отменяет отложенное удаление.
    _touch(conversation, ts)
    return {"conversationId": cid, "turns": conversation.turns}


@app.delete("/api/conversations/{cid}", status_code=204)
def delete_conversation(cid: str) -> Response:
    _HISTORY.pop(cid, None)
    _CHAT_LOCKS.pop(cid, None)
    return Response(status_code=204)


@app.post("/api/conversations/{cid}/close")
def close_conversation(cid: str, ts: int | None = None) -> dict:
    conversation = _HISTORY.get(cid)
    if conversation is None:
        return {"status": "ok"}
    # beacon мог прийти уже после запроса перезагруженной страницы:
    # тогда его метка времени старше последней активности — игнорируем.
    if ts is not None and ts < conversation.last_seen_ms:
        return {"status": "stale"}
    conversation.closing_at = time.monotonic()
    return {"status": "ok"}
