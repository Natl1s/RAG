"""Тесты FastAPI-заглушки."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_chat_stream_returns_fixed_answer() -> None:
    with client.stream("POST", "/api/chat", json={"query": "Любой вопрос"}) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        body = "".join(res.iter_text())

    assert '"type": "token"' in body
    assert '"type": "done"' in body
    assert "BLOB" in body
