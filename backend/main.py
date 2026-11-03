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

KB_PATH = Path(__file__).resolve().parents[1] / "frontend" / "knowledge" / "knowledge.json"
with open(KB_PATH, encoding="utf-8") as fh:
    KB = json.load(fh)

CHUNKS = KB["chunks"]
TOPICS = KB["topics"]

STOP_WORDS = {
    "как", "что", "какой", "какая", "какое", "какие", "каков", "какова", "каковы",
    "кто", "где", "когда", "почему", "зачем", "отчего", "сколько", "чей", "чья", "чьё", "чьи",
    "нужно", "надо", "необходимо", "можно", "могу", "можешь", "может", "можем", "могут", "следует", "стоит",
    "для", "при", "под", "над", "перед", "после", "между", "через", "вокруг", "около", "без", "вне", "внутри",
    "это", "этот", "эта", "эти", "этого", "этому", "этим", "этом", "тот", "та", "то", "те",
    "такой", "такая", "такое", "такие",
    "есть", "если", "или", "либо", "и", "а", "но", "да", "же", "ли", "бы", "то", "так",
    "вот", "вон", "уже", "ещё", "тоже", "также", "только", "просто", "вообще",
    "в", "во", "на", "с", "со", "от", "ото", "до", "по", "за", "из", "изо", "о", "об", "обо",
    "у", "к", "ко", "не", "ни", "нет",
    "мне", "меня", "мой", "моя", "моё", "мои", "моего", "моей", "моим", "моем",
    "вы", "вам", "вас", "ваш", "ваша", "ваше", "ваши",
    "я", "он", "она", "оно", "они", "мы", "ты", "себе", "себя", "свой", "своя", "своё", "свои",
    "весь", "вся", "всё", "все", "всего", "всей", "всем", "буду", "будет", "будут", "быть",
    "был", "была", "были", "было",
    "данных", "данные", "данным", "данными", "данного", "данной",
    # Обобщённые единицы времени и денег — не должны вызывать кросс-совпадения.
    "дней", "день", "дня", "дни", "лет", "год", "года", "годов", "месяц", "месяца", "месяцев",
    "неделя", "недели", "недель", "тысяч", "тысячи", "тысячу", "млн", "тыс", "руб", "рубля", "рублей",
}

RETRIEVAL_MIN_SCORE = 2
ANSWER_MIN_SCORE = 3
MAX_SOURCES = 3

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


def _stem(word: str) -> str:
    """Грубый стеммер: срезает частые русские окончания («согласие»/«согласия» -> «согласи»)."""
    endings = (
        "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими",
        "ая", "яя", "ое", "ее", "ые", "ие", "ой", "ей", "ый", "ий",
        "ых", "их", "ым", "им", "ом", "ем", "ов", "ев", "ам", "ям", "ах", "ях",
        "а", "я", "о", "е", "у", "ю", "ы", "и", "ь", "й",
    )
    for e in endings:
        if len(word) - len(e) >= 3 and word.endswith(e):
            return word[: -len(e)]
    return word


def tokenize(text: str) -> list[str]:
    text = re.sub(r"[^a-zа-яё0-9\s-]", " ", text.lower())
    return [_stem(t) for t in re.split(r"[\s-]+", text) if len(t) >= 2 and t not in STOP_WORDS]


def normalize(score: int) -> float:
    return round(min(1.0, score / 8.0), 2)


def run_retrieval(query: str):
    q_tokens = tokenize(query)
    q_lower = query.lower()

    scored = []
    for chunk in CHUNKS:
        score = 0
        for kw in chunk["keywords"]:
            kl = kw.lower()
            is_phrase = " " in kl or "-" in kl
            if is_phrase and (kl in q_lower or q_lower in kl):
                score += 4

        kw_tokens = set()
        for kw in chunk["keywords"]:
            kw_tokens.update(tokenize(kw))
        text_tokens = set(tokenize(chunk["text"]))

        for t in q_tokens:
            if t in kw_tokens:
                score += 3
            elif t in text_tokens:
                score += 1

        scored.append((score, chunk))

    scored.sort(key=lambda pair: -pair[0])
    best_raw = scored[0][0] if scored else 0

    sources = []
    for score, chunk in scored[:MAX_SOURCES]:
        if score >= RETRIEVAL_MIN_SCORE:
            sources.append({**chunk, "score": normalize(score)})

    return sources, normalize(best_raw), best_raw >= ANSWER_MIN_SCORE


def not_found_answer(query: str, sources: list[dict]) -> str:
    related = ""
    if sources:
        lines = "\n".join(f"- {s['cite']} — {s['summary']}" for s in sources)
        related = f"Ближайшие темы в базе знаний:\n{lines}\n\n"
    topics = "В базе знаний доступны разделы:\n" + "\n".join(f"- {t}" for t in TOPICS) + "\n\n"
    return (
        f"К сожалению, в базе знаний пока нет ответа на ваш вопрос: «{query}».\n\n"
        f"{related}{topics}"
        "Попробуйте переформулировать вопрос или уточнить детали (закон, тип документа, ситуацию). "
        "Либо нажмите «Передать вопрос юристу», и специалист ПСБ свяжется с вами."
    )


def generate_answer(query: str, sources: list[dict], answered: bool) -> str:
    if not answered:
        return not_found_answer(query, sources)

    primary = sources[0]
    secondary = [s for s in sources[1:] if s["score"] >= 0.5]
    if not secondary:
        return primary["answer"]

    extra = "\n".join(f"- {s['cite']}: {s['summary']}" for s in secondary)
    return f"{primary['answer']}\n\nДополнительно по теме:\n{extra}"


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def stream(query: str) -> AsyncGenerator[str, None]:
    start = time.time()

    # Имитация задержки ретривера.
    await asyncio.sleep(0.32 + random.random() * 0.48)

    sources, confidence, answered = run_retrieval(query)
    status = "answered" if answered else "not_found"
    yield sse({"type": "retrieval", "sources": sources, "confidence": confidence, "status": status})

    answer = generate_answer(query, sources, answered)
    for token in re.findall(r"\S+\s*", answer):
        yield sse({"type": "token", "text": token})
        await asyncio.sleep(0.008 + random.random() * 0.022)

    latency_ms = int((time.time() - start) * 1000)
    yield sse({
        "type": "done",
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "status": status,
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
    return {"status": "ok", "mode": "stub", "chunks": len(CHUNKS)}


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
