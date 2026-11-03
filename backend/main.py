"""FastAPI-заглушка конвейера «RAG + LLM» для юридического ассистента ПСБ.

Здесь имитируется поведение реального бэкенда:
  1. retrieve()  — заглушка RAG-ретривера (лексическое ранжирование по чанкам);
  2. generate()  — заглушка LLM (сборка ответа или «ответа нет в базе»);
  3. /api/chat   — поток Server-Sent Events (retrieval -> token* -> done).

База знаний читается из frontend/knowledge/knowledge.json, поэтому фронтенд-заглушка
и этот бэкенд отвечают одинаково.

Запуск:  uvicorn main:app --reload --port 8000
"""

import asyncio
import json
import random
import re
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="PSB Legal Assistant API (stub)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str

class EscalateRequest(BaseModel):
    query: str
    sources: list[dict] = Field(default_factory=list)

class EscalationResponse(BaseModel):
    ticketId: str
    status: str = "created"
    message: str

def generate_answer(query: str) -> str:
    return "BLOB"


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def stream(query: str) -> AsyncGenerator[str, None]:
    start = time.time()

    # Имитация задержки ретривера.
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


@app.post("/api/escalate", response_model=EscalationResponse)
def escalate(req: EscalateRequest) -> EscalationResponse:
    """Заглушка создания заявки юристу. В реальной системе — интеграция с CRM."""
    ticket = f"PSB-{int(time.time() * 1000) % 1_000_000:06d}"
    return EscalationResponse(
        ticketId=ticket,
        message="Вопрос передан юристу ПСБ. Специалист свяжется с вами в течение 1 рабочего дня.",
    )


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
