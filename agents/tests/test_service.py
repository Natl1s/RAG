from __future__ import annotations

import asyncio
import unittest

from agents.service import AgentService
from rag.index.contracts import RetrievedChunk


class FakeRetriever:
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                doc_id="doc-1",
                chunk_id="chunk-1",
                text="Текст документа",
                metadata={
                    "title": "Заголовок",
                    "doc_type": "Закон",
                    "docdate": "01.01.2024",
                },
                score=0.9,
            )
        ]


class AgentServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AgentService({"hybrid": FakeRetriever()})

    def test_sync_answer_returns_rag_context(self) -> None:
        answer = self.service.answer("u-1", "налоги")

        self.assertIn("[Источник 1]", answer)
        self.assertIn("text: Текст документа", answer)

    def test_async_answer_returns_sources(self) -> None:
        result = asyncio.run(self.service.aanswer("u-1", "налоги"))

        self.assertIn("text: Текст документа", result.answer)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0]["doc_id"], "doc-1")
        self.assertEqual(result.sources[0]["chunk_id"], "chunk-1")

    def test_memory_keeps_thread_history(self) -> None:
        asyncio.run(self.service.aanswer("u-1", "первый"))
        asyncio.run(self.service.aanswer("u-1", "второй"))

        state = asyncio.run(
            self.service._graph.aget_state({"configurable": {"thread_id": "u-1"}})
        )

        self.assertEqual(
            [message.type for message in state.values["messages"]],
            ["human", "ai", "human", "ai"],
        )

    def test_empty_question_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.service.answer("u-1", "   ")

    def test_empty_thread_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.service.answer("", "вопрос")

    def test_sync_answer_inside_event_loop_raises(self) -> None:
        async def call_sync() -> None:
            with self.assertRaises(RuntimeError):
                self.service.answer("u-1", "вопрос")

        asyncio.run(call_sync())


if __name__ == "__main__":
    unittest.main()
