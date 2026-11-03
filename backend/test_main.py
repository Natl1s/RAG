"""Тесты FastAPI-заглушки: ретривер, поток SSE и эскалация."""

from fastapi.testclient import TestClient

from main import app, run_retrieval

client = TestClient(app)


def test_health() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["mode"] == "stub"
    assert data["chunks"] > 0


def test_retrieval_finds_known_topic() -> None:
    sources, confidence, answered = run_retrieval("Как вернуть товар в течение 14 дней?")
    assert answered is True
    assert confidence > 0
    assert sources
    assert sources[0]["cite"] == "ЗоЗПП, ст. 25"


def test_retrieval_reports_missing_topic() -> None:
    sources, _, answered = run_retrieval("Какие документы нужны для регистрации ООО?")
    assert answered is False


def test_chat_stream_answered() -> None:
    with client.stream("POST", "/api/chat", json={"query": "Сколько длится ежегодный отпуск?"}) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        body = "".join(res.iter_text())

    assert '"type": "retrieval"' in body
    assert '"type": "done"' in body
    assert '"status": "answered"' in body
    assert "28 календарных дней" in body


def test_chat_stream_not_found() -> None:
    with client.stream("POST", "/api/chat", json={"query": "Какие документы нужны для регистрации ООО?"}) as res:
        body = "".join(res.iter_text())

    assert '"status": "not_found"' in body
    assert "Передать вопрос юристу" in body


def test_escalation_creates_ticket() -> None:
    res = client.post(
        "/api/escalate",
        json={"query": "Какие документы нужны для регистрации ООО?", "sources": []},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "created"
    assert data["ticketId"].startswith("PSB-")
    assert "юристу" in data["message"]
