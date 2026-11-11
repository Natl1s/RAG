from __future__ import annotations

import asyncio
import unittest

from agents.config import Settings
from agents.service import AgentService


class FakeSearch:
    def __init__(self, context: str) -> None:
        self.context = context
        self.calls: list[tuple[str, object]] = []

    def __call__(self, query: str, retrievers: object) -> str:
        self.calls.append((query, retrievers))
        return self.context


class AgentServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.retrievers = object()
        self.search = FakeSearch("Источник: doc-1\nТекст: Текст документа")
        self.service = AgentService(
            self.retrievers,
            settings=Settings(llm_base_url="", llm_api_key="", llm_model=""),
            search_fn=self.search,
        )

    def test_sync_answer_returns_rag_context(self) -> None:
        answer = self.service.answer("u-1", "налоги")

        self.assertIn("Источник: doc-1", answer)
        self.assertIn("Текст: Текст документа", answer)

    def test_async_answer_returns_result(self) -> None:
        result = asyncio.run(self.service.aanswer("u-1", "налоги"))

        self.assertIn("Текст: Текст документа", result.answer)

    def test_search_receives_question_and_retrievers(self) -> None:
        asyncio.run(self.service.aanswer("u-1", "налоги"))

        self.assertEqual(self.search.calls, [("налоги", self.retrievers)])

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
