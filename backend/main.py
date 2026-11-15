"""FastAPI-заглушка чат-бота «Юридический ассистент ПСБ».

RAG и LLM заменены одной заглушкой: на любой вопрос возвращается одна и та же
строка. Поток отдаётся как Server-Sent Events: token* -> done.

Запуск:  uvicorn main:app --reload --port 8000
"""

import asyncio
import json
import random
import re
import time
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag.config import RAGConfig
from rag.rag_pipeline import RAG
from pathlib import Path
import pandas as pd
from agents.bootstrap import build_local_service
RETRIEVERS: dict[str, object] | None = None

def warm_up(app: FastAPI) -> None:
    docs = "./dataset/result_chunks.csv"
    docs = pd.read_csv(docs)
    rag_client = RAG(RAGConfig())
    rag_client.build(docs)
    global AGENT
    if AGENT is not None:
        return
    AGENT = build_local_service(rag_client)

@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_up(app)
    yield

app = FastAPI(title="PSB Legal Assistant API (stub)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str


def generate_answer(query: str) -> str:
    return AGENT.answer('1', query)


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def stream(query: str) -> AsyncGenerator[str, None]:
    start = time.time()

    # Имитация задержки генерации.
    await asyncio.sleep(0.32 + random.random() * 0.48)

    answer = generate_answer(query)
    for token in re.findall(r"\S+\s*", answer):
        yield sse({"type": "token", "text": token})
        await asyncio.sleep(0.008 + random.random() * 0.022)

    latency_ms = int((time.time() - start) * 1000)
    yield sse({
        "type": "done",
        "answer": answer,
        "latencyMs": latency_ms,
    })
    yield "data: [DONE]\n\n"


@app.get("/")
def root() -> dict:
    return {
        "service": "PSB Legal Assistant API (stub)",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream(req.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
