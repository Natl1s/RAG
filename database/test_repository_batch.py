"""Проверки транзакций без PostgreSQL: DATABASE_URL=sqlite:// pytest database/test_repository_batch.py."""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.create_tables import ChunkModel
from database.repository import ChunkRepository


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    # PostgreSQL-типы модели сериализуются ORM; для проверки транзакций
    # достаточно совместимой таблицы SQLite без векторного поиска.
    with engine.begin() as connection:
        connection.exec_driver_sql(
            'CREATE TABLE chunks (id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL, '
            'text TEXT NOT NULL, metadata TEXT NOT NULL, embedding TEXT NOT NULL)'
        )
    with Session(engine, expire_on_commit=False) as session:
        yield session
    engine.dispose()


def make_chunk(doc_id="document", text="text"):
    return ChunkModel(
        doc_id=doc_id,
        text=text,
        metadata_={"article": "70"},
        embedding=[0.1] * 384,
    )


def test_save_chunks_accepts_generator_and_persists_batch(session):
    chunks = [make_chunk("first"), make_chunk("second")]

    result = ChunkRepository(session).save_chunks(chunk for chunk in chunks)

    assert result == chunks
    assert all(chunk.id is not None for chunk in result)
    session.expunge_all()
    saved = session.scalars(select(ChunkModel).order_by(ChunkModel.id)).all()
    assert [chunk.doc_id for chunk in saved] == ["first", "second"]
    assert saved[0].metadata_ == {"article": "70"}
    assert len(saved[0].embedding) == 384


def test_failed_batch_is_atomic_and_session_can_be_reused(session):
    repository = ChunkRepository(session)
    existing = repository.save_chunks([make_chunk("existing")])[0]

    with pytest.raises(IntegrityError):
        repository.save_chunks([make_chunk("valid"), make_chunk(text=None)])

    assert session.scalars(select(ChunkModel.id)).all() == [existing.id]
    repository.save_chunks([make_chunk("after_failure")])
    assert len(repository.get_chunks()) == 2


def test_empty_batch_does_not_commit_pending_changes(session):
    pending = make_chunk()
    session.add(pending)

    assert ChunkRepository(session).save_chunks([]) == []
    assert pending in session.new
    session.rollback()
    assert ChunkRepository(session).get_chunks() == []
