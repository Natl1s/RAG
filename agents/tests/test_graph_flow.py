from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from agents.config import Settings
from agents.nodes import route_after_grade, route_after_verify
from agents.service import AgentService
from agents.state import (
    ClarificationRequest,
    ContextChunk,
    PremiseCheck,
    RetrievalGradeResult,
    SubQuery,
    SubqueryGrade,
    UnderstandResult,
    VerificationResult,
)
from rag.index.contracts import RetrievedChunk


class FakeRAG:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search_chunks(self, query: str) -> list[RetrievedChunk]:
        self.calls.append(query)
        return [
            RetrievedChunk(
                doc_id="doc-1",
                chunk_id="chunk-1",
                text="Текст про налоги.",
                metadata={"title": "Документ"},
                score=0.9,
            )
        ]


class TwoContractRAG(FakeRAG):
    def search_chunks(self, query: str) -> list[RetrievedChunk]:
        self.calls.append(query)
        return [
            RetrievedChunk(
                doc_id="документы компании__Договор поставки П-17-2024 Альфа-Трейд.txt",
                chunk_id="alfa-1",
                text="Гарантийный срок — 24 месяца.",
                metadata={
                    "title": "документы компании__Договор поставки П-17-2024 Альфа-Трейд.txt"
                },
                score=0.9,
            ),
            RetrievedChunk(
                doc_id="документы компании__Договор поставки ПТ-05-2023 Альфа-Трейдинг.txt",
                chunk_id="trading-1",
                text="Гарантийный срок — 12 месяцев.",
                metadata={
                    "title": "документы компании__Договор поставки ПТ-05-2023 Альфа-Трейдинг.txt"
                },
                score=0.8,
            ),
        ]


def make_understand(in_scope: bool = True) -> UnderstandResult:
    subquery = SubQuery(id=1, text="налоги", part_ids=[1], kind="fact")
    return UnderstandResult(
        standalone_query="налоги",
        in_scope=in_scope,
        scope_reason="тест",
        is_complex=False,
        premise_check=PremiseCheck(),
        subqueries=[subquery],
    )


def make_grade(coverage: str) -> RetrievalGradeResult:
    return RetrievalGradeResult(
        subqueries=[
            SubqueryGrade(
                subquery_id=1,
                coverage=coverage,
                missing_aspects=[] if coverage == "full" else ["нет данных"],
            )
        ],
        overall=coverage,
        missing_aspects=[] if coverage == "full" else ["нет данных"],
    )


def make_verify(verdict: str) -> VerificationResult:
    return VerificationResult(
        verdict=verdict,
        grounded=verdict == "pass",
        answers_question=True,
        citation_ok=True,
        premise_ok=True,
        failure_type="none" if verdict == "pass" else "retrieval",
    )


class GraphFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rag = FakeRAG()
        self.settings = Settings(
            llm_base_url="http://llm.local/v1",
            llm_api_key="test-key",
            llm_model="test-model",
        )

    def _patch(self, name: str, value: object) -> None:
        patcher = patch(f"agents.nodes.{name}", new=value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_out_of_scope_does_not_retrieve(self) -> None:
        self._patch("understand_query", AsyncMock(return_value=make_understand(False)))
        service = AgentService(self.rag, settings=self.settings)

        answer = asyncio.run(service.aanswer("u-1", "кто выиграет чемпионат?"))

        self.assertIn("выходит за тематику", answer.answer)
        self.assertEqual(self.rag.calls, [])

    def test_clarification_before_retrieval(self) -> None:
        understand = make_understand()
        understand.clarification = ClarificationRequest(
            needed=True,
            question="Уточните, пожалуйста, номер договора.",
            missing_slots=["номер договора"],
        )
        self._patch("understand_query", AsyncMock(return_value=understand))
        service = AgentService(self.rag, settings=self.settings)

        result = asyncio.run(
            service.aanswer(
                "u-1",
                "Мы арендуем склад у «Северного Склада». Планируем съезжать.",
            )
        )

        self.assertEqual(result.status, "clarification")
        self.assertIn("номер договора", result.answer)
        self.assertIsNotNone(result.clarification)
        self.assertEqual(self.rag.calls, [])

    def test_clarification_is_not_repeated_before_retrieval(self) -> None:
        understand = make_understand()
        understand.clarification = ClarificationRequest(
            needed=True,
            question="Уточните, пожалуйста, номер договора.",
        )
        self._patch("understand_query", AsyncMock(return_value=understand))
        self._patch("grade_retrieval", AsyncMock(return_value=make_grade("full")))
        self._patch(
            "generate_answer",
            AsyncMock(return_value="Ответ [doc_id: doc-1]"),
        )
        self._patch("verify_answer", AsyncMock(return_value=make_verify("pass")))
        service = AgentService(self.rag, settings=self.settings)

        result = asyncio.run(
            service.aanswer(
                "u-1",
                "Пользователь: Мы арендуем склад. Планируем съезжать.\n"
                "Ассистент: Уточните номер договора?\n"
                "Пользователь: Не знаю.",
            )
        )

        self.assertNotEqual(result.status, "clarification")
        self.assertEqual(self.rag.calls, ["налоги"])

    def test_clarification_after_retrieval_ambiguity(self) -> None:
        rag = TwoContractRAG()
        understand = make_understand()
        understand.standalone_query = "Подскажите гарантию по договору с «Альфой»."
        understand.subqueries = [
            SubQuery(
                id=1,
                text=understand.standalone_query,
                part_ids=[1],
                kind="fact",
            )
        ]
        generate = AsyncMock(return_value="Не должно вызываться.")
        self._patch("understand_query", AsyncMock(return_value=understand))
        self._patch("grade_retrieval", AsyncMock(return_value=make_grade("full")))
        self._patch("generate_answer", generate)
        service = AgentService(rag, settings=self.settings)

        result = asyncio.run(
            service.aanswer("u-1", "Подскажите гарантию по договору с «Альфой».")
        )

        self.assertEqual(result.status, "clarification")
        self.assertIn("Альфа-Трейд", result.answer)
        self.assertIn("Альфа-Трейдинг", result.answer)
        self.assertEqual(rag.calls, ["Подскажите гарантию по договору с «Альфой»."])
        generate.assert_not_awaited()

    def test_retrieval_rewrite_cycle(self) -> None:
        self._patch("understand_query", AsyncMock(return_value=make_understand()))
        self._patch(
            "grade_retrieval",
            AsyncMock(side_effect=[make_grade("none"), make_grade("full")]),
        )
        self._patch("rewrite_subquery", AsyncMock(return_value="налоги и сборы"))
        self._patch(
            "generate_answer",
            AsyncMock(return_value="Ответ [doc_id: doc-1]"),
        )
        self._patch("verify_answer", AsyncMock(return_value=make_verify("pass")))
        service = AgentService(self.rag, settings=self.settings)

        answer = asyncio.run(service.aanswer("u-1", "налоги"))

        self.assertIn("Ответ", answer.answer)
        self.assertEqual(self.rag.calls, ["налоги", "налоги и сборы"])

    def test_answer_revision_cycle(self) -> None:
        self._patch("understand_query", AsyncMock(return_value=make_understand()))
        self._patch("grade_retrieval", AsyncMock(return_value=make_grade("full")))
        self._patch(
            "generate_answer",
            AsyncMock(return_value="Черновик [doc_id: doc-1]"),
        )
        self._patch(
            "verify_answer",
            AsyncMock(
                side_effect=[
                    make_verify("revise_answer"),
                    make_verify("pass"),
                ]
            ),
        )
        revise = AsyncMock(return_value="Исправленный ответ [doc_id: doc-1]")
        self._patch("revise_answer", revise)
        service = AgentService(self.rag, settings=self.settings)

        answer = asyncio.run(service.aanswer("u-1", "налоги"))

        self.assertIn("Исправленный ответ", answer.answer)
        revise.assert_awaited_once()

    def test_rewrite_is_limited_per_round(self) -> None:
        subqueries = [
            SubQuery(
                id=index,
                text=f"подзапрос {index}",
                part_ids=[1],
                kind="fact",
            )
            for index in range(1, 5)
        ]
        understand = UnderstandResult(
            standalone_query="сложный вопрос",
            in_scope=True,
            scope_reason="тест",
            is_complex=True,
            premise_check=PremiseCheck(),
            subqueries=subqueries,
        )
        self._patch("understand_query", AsyncMock(return_value=understand))
        self._patch(
            "grade_retrieval",
            AsyncMock(side_effect=[make_grade("none"), make_grade("full")]),
        )
        rewrite = AsyncMock(return_value="новый запрос")
        self._patch("rewrite_subquery", rewrite)
        self._patch(
            "generate_answer",
            AsyncMock(return_value="Ответ [doc_id: doc-1]"),
        )
        self._patch("verify_answer", AsyncMock(return_value=make_verify("pass")))
        settings = Settings(
            llm_base_url="http://llm.local/v1",
            llm_api_key="test-key",
            llm_model="test-model",
            max_rewrite_subqueries_per_round=2,
        )
        service = AgentService(self.rag, settings=settings)

        asyncio.run(service.aanswer("u-1", "сложный вопрос"))

        self.assertEqual(rewrite.await_count, 2)

    def test_soft_budget_skips_optional_rewrite(self) -> None:
        state = {
            "coverage": "partial",
            "total_steps": 2,
            "retrieval_attempts": 1,
            "used_tokens": self.settings.max_total_tokens_soft,
        }

        self.assertEqual(route_after_grade(state, self.settings), "generate")

        state["coverage"] = "none"
        self.assertEqual(route_after_grade(state, self.settings), "finalize")

    def test_soft_budget_skips_revision(self) -> None:
        state = {
            "verification": make_verify("revise_answer"),
            "total_steps": 5,
            "answer_attempts": 1,
            "retrieval_attempts": 1,
            "used_tokens": self.settings.max_total_tokens_soft,
        }

        self.assertEqual(route_after_verify(state, self.settings), "finalize")

    def test_hard_budget_prevents_llm_calls(self) -> None:
        builder = patch(
            "agents.llm._build_llm",
            side_effect=AssertionError("real LLM must not be called"),
        )
        builder.start()
        self.addCleanup(builder.stop)
        settings = Settings(
            llm_base_url="http://llm.local/v1",
            llm_api_key="test-key",
            llm_model="test-model",
            max_total_tokens_per_question=100,
            max_total_tokens_soft=50,
        )
        service = AgentService(self.rag, settings=settings)

        answer = asyncio.run(service.aanswer("u-1", "налоги"))

        self.assertIn("нет документов", answer.answer)

    def test_no_data_refuses_after_retrieval_limit(self) -> None:
        self._patch("understand_query", AsyncMock(return_value=make_understand()))
        self._patch(
            "grade_retrieval",
            AsyncMock(side_effect=[make_grade("none"), make_grade("none")]),
        )
        self._patch("rewrite_subquery", AsyncMock(return_value="другой запрос"))
        service = AgentService(self.rag, settings=self.settings)

        answer = asyncio.run(service.aanswer("u-1", "неизвестный вопрос"))

        self.assertIn("нет документов", answer.answer)
        self.assertEqual(len(self.rag.calls), 2)


if __name__ == "__main__":
    unittest.main()
