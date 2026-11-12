from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import SystemMessage

from agents.config import Settings
from agents.llm import (
    _chat_completion,
    generate_answer,
    grade_retrieval,
    revise_answer,
    rewrite_subquery,
    understand_query,
    verify_answer,
)
from agents.state import ContextChunk, DialogTurn, PremiseCheck, SubQuery


def make_settings(**overrides) -> Settings:
    data = {
        "llm_base_url": "http://llm.local/v1",
        "llm_api_key": "test-key",
        "llm_model": "test-model",
        "temperature": 0.0,
    }
    data.update(overrides)
    return Settings(**data)


def make_context(subquery_id: int = 1) -> ContextChunk:
    return ContextChunk(
        doc_id="doc-1",
        chunk_id="chunk-1",
        text="Сотрудник обязан уплатить налог до 1 декабря.",
        metadata={"title": "Налоговый кодекс"},
        score=0.9,
        subquery_ids=[subquery_id],
    )


def make_subquery(text: str = "налоги", subquery_id: int = 1) -> SubQuery:
    return SubQuery(id=subquery_id, text=text, part_ids=[1], kind="fact")


class UnderstandQueryTest(unittest.IsolatedAsyncioTestCase):
    async def test_parses_and_normalizes_complex_question(self) -> None:
        raw = (
            '{"standalone_query":"Какие налоги платит ИП по договору № 1?",'
            '"in_scope":true,"scope_reason":"налоги","is_complex":true,'
            '"question_parts":[{"id":1,"text":"налоги","kind":"fact"}],'
            '"premise_check":{"has_false_premise":true,'
            '"false_premise":"ИП не платит налоги",'
            '"explanation":"ИП платит"},'
            '"subqueries":['
            '{"id":1,"text":"налоги ИП","part_ids":[1],"kind":"fact"},'
            '{"id":2,"text":"срок уплаты налогов ИП","part_ids":[1],"kind":"fact"}]}'
        )
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value=raw)):
            result = await understand_query(
                [DialogTurn(role="user", text="Я ИП")],
                "Какие налоги?",
                settings=make_settings(),
            )

        self.assertEqual(result.standalone_query, "Какие налоги платит ИП по договору № 1?")
        self.assertTrue(result.is_complex)
        self.assertEqual(len(result.subqueries), 2)
        self.assertTrue(result.premise_check.has_false_premise)

    async def test_simple_question_keeps_one_subquery(self) -> None:
        raw = (
            '{"standalone_query":"налоги","in_scope":true,'
            '"is_complex":false,"subqueries":['
            '{"id":1,"text":"налоги","part_ids":[1]},'
            '{"id":2,"text":"штрафы","part_ids":[1]}]}'
        )
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value=raw)):
            result = await understand_query([], "налоги", settings=make_settings())

        self.assertEqual(len(result.subqueries), 1)

    async def test_parses_clarification_request(self) -> None:
        raw = (
            '{"standalone_query":"гарантия по договору с Альфой",'
            '"in_scope":true,"is_complex":false,'
            '"clarification":{"needed":true,'
            '"question":"Уточните, какая именно компания имеется в виду.",'
            '"missing_slots":["компания"],"options":["Альфа-Трейд","Альфа-Трейдинг"],'
            '"reason":"неоднозначная компания"},'
            '"subqueries":[{"id":1,"text":"гарантия Альфа",'
            '"part_ids":[1],"kind":"fact"}]}'
        )
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value=raw)):
            result = await understand_query(
                [],
                "Подскажите гарантию по договору с Альфой.",
                settings=make_settings(),
            )

        self.assertTrue(result.clarification.needed)
        self.assertIn("какая именно компания", result.clarification.question)
        self.assertEqual(len(result.clarification.options), 2)

    async def test_builds_clarification_question_from_missing_slots(self) -> None:
        raw = (
            '{"standalone_query":"возврат помещения","in_scope":true,'
            '"is_complex":false,'
            '"clarification":{"needed":true,"question":"",'
            '"missing_slots":["номер договора"]},'
            '"subqueries":[{"id":1,"text":"возврат помещения",'
            '"part_ids":[1],"kind":"fact"}]}'
        )
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value=raw)):
            result = await understand_query(
                [],
                "Мы арендуем склад. Планируем съезжать.",
                settings=make_settings(),
            )

        self.assertTrue(result.clarification.needed)
        self.assertIn("номер договора", result.clarification.question)

    async def test_falls_back_to_last_question_on_bad_json(self) -> None:
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value="не json")):
            result = await understand_query([], "налоги", settings=make_settings())

        self.assertEqual(result.standalone_query, "налоги")
        self.assertEqual(result.subqueries[0].text, "налоги")
        self.assertFalse(result.clarification.needed)


class GradeRetrievalTest(unittest.IsolatedAsyncioTestCase):
    async def test_parses_subquery_grades_and_computes_overall(self) -> None:
        raw = (
            '{"subqueries":[{"subquery_id":1,"coverage":"partial",'
            '"missing_aspects":["срок"],"reason":"есть норма"}],'
            '"overall":"full","missing_aspects":[]}'
        )
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value=raw)):
            result = await grade_retrieval(
                "налоги",
                [make_subquery()],
                [make_context()],
                settings=make_settings(),
            )

        self.assertEqual(result.overall, "partial")
        self.assertEqual(result.subqueries[0].missing_aspects, ["срок"])

    async def test_empty_context_does_not_call_llm(self) -> None:
        completion = AsyncMock(return_value="")
        with patch("agents.llm._chat_completion", new=completion):
            result = await grade_retrieval(
                "налоги",
                [make_subquery()],
                [],
                settings=make_settings(),
            )

        self.assertEqual(result.overall, "none")
        completion.assert_not_awaited()

    async def test_missing_subquery_grade_uses_heuristic(self) -> None:
        raw = '{"subqueries":[],"overall":"none","missing_aspects":[]}'
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value=raw)):
            result = await grade_retrieval(
                "налоги",
                [make_subquery()],
                [make_context()],
                settings=make_settings(),
            )

        self.assertEqual(result.overall, "full")


class RewriteSubqueryTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_rewritten_query(self) -> None:
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value="новый запрос")):
            result = await rewrite_subquery(
                "вопрос",
                make_subquery("старый запрос"),
                "нужно точнее",
                [],
                settings=make_settings(),
            )

        self.assertEqual(result, "новый запрос")

    async def test_falls_back_to_current_query(self) -> None:
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value="")):
            result = await rewrite_subquery(
                "вопрос",
                make_subquery("старый запрос"),
                "нужно точнее",
                [],
                settings=make_settings(),
            )

        self.assertEqual(result, "старый запрос")

    async def test_avoids_near_duplicate_of_tried_query(self) -> None:
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value="старый запрос")):
            result = await rewrite_subquery(
                "вопрос",
                make_subquery("старый запрос"),
                "нужно точнее",
                ["старый запрос"],
                settings=make_settings(),
            )

        self.assertEqual(result, "старый запрос")


class GenerateAnswerTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_llm_answer_when_available(self) -> None:
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value="Ответ LLM")):
            answer = await generate_answer(
                "вопрос",
                [make_context()],
                [],
                PremiseCheck(),
                "full",
                [],
                settings=make_settings(),
            )

        self.assertEqual(answer, "Ответ LLM")

    async def test_falls_back_to_context_when_llm_empty(self) -> None:
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value="")):
            answer = await generate_answer(
                "вопрос",
                [make_context()],
                [],
                PremiseCheck(),
                "full",
                [],
                settings=make_settings(),
            )

        self.assertIn("doc-1", answer)
        self.assertIn("налог", answer)

    async def test_empty_context_does_not_call_llm(self) -> None:
        completion = AsyncMock(return_value="Ответ LLM")
        with patch("agents.llm._chat_completion", new=completion):
            answer = await generate_answer(
                "вопрос",
                [],
                [],
                PremiseCheck(),
                "none",
                [],
                settings=make_settings(),
            )

        self.assertIn("нет документов", answer)
        completion.assert_not_awaited()


class VerifyAnswerTest(unittest.IsolatedAsyncioTestCase):
    async def test_parses_valid_verification(self) -> None:
        raw = (
            '{"verdict":"pass","grounded":true,"answers_question":true,'
            '"citation_ok":true,"premise_ok":true,"failure_type":"none",'
            '"missing_evidence":[],"fix_hint":"","reason":"ok"}'
        )
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value=raw)):
            result = await verify_answer(
                "вопрос",
                "Ответ [doc_id: doc-1]",
                [make_context()],
                PremiseCheck(),
                settings=make_settings(),
            )

        self.assertEqual(result.verdict, "pass")

    async def test_inconsistent_pass_becomes_revision(self) -> None:
        raw = (
            '{"verdict":"pass","grounded":false,"answers_question":true,'
            '"citation_ok":true,"premise_ok":true,"failure_type":"generation",'
            '"missing_evidence":[],"fix_hint":"убрать выдумку","reason":"не заземлён"}'
        )
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value=raw)):
            result = await verify_answer(
                "вопрос",
                "Ответ",
                [make_context()],
                PremiseCheck(),
                settings=make_settings(),
            )

        self.assertEqual(result.verdict, "revise_answer")

    async def test_bad_citation_becomes_revision_on_fallback(self) -> None:
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value="не json")):
            result = await verify_answer(
                "вопрос",
                "Ответ [doc_id: doc-999]",
                [make_context()],
                PremiseCheck(),
                settings=make_settings(),
            )

        self.assertEqual(result.verdict, "revise_answer")
        self.assertEqual(result.failure_type, "citation")

    async def test_empty_context_refuses_without_llm(self) -> None:
        completion = AsyncMock(return_value="")
        with patch("agents.llm._chat_completion", new=completion):
            result = await verify_answer(
                "вопрос",
                "Ответ",
                [],
                PremiseCheck(),
                settings=make_settings(),
            )

        self.assertEqual(result.verdict, "refuse")
        completion.assert_not_awaited()


class ReviseAnswerTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_revised_answer(self) -> None:
        with patch("agents.llm._chat_completion", new=AsyncMock(return_value="Исправлено")):
            result = await revise_answer(
                "вопрос",
                "Черновик",
                [make_context()],
                None,
                settings=make_settings(),
            )

        self.assertEqual(result, "Исправлено")

    async def test_empty_context_returns_no_data_answer(self) -> None:
        completion = AsyncMock(return_value="Исправлено")
        with patch("agents.llm._chat_completion", new=completion):
            result = await revise_answer(
                "вопрос",
                "Черновик",
                [],
                None,
                settings=make_settings(),
            )

        self.assertIn("нет документов", result)
        completion.assert_not_awaited()


class ContextLimitTest(unittest.IsolatedAsyncioTestCase):
    def _context(self, text: str) -> ContextChunk:
        return ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text=text,
            metadata={},
            subquery_ids=[1],
        )

    async def test_grade_uses_grade_context_limit(self) -> None:
        raw = '{"subqueries":[],"overall":"none","missing_aspects":[]}'
        completion = AsyncMock(return_value=raw)
        chunk = self._context("Начало " + "Т" * 500 + " ХВОСТ")

        with patch("agents.llm._chat_completion", new=completion):
            await grade_retrieval(
                "вопрос",
                [make_subquery()],
                [chunk],
                settings=make_settings(max_grade_context_chars=100),
            )

        messages = completion.await_args.kwargs["messages"]
        self.assertNotIn("ХВОСТ", messages[1].content)

    async def test_generate_uses_answer_context_limit(self) -> None:
        completion = AsyncMock(return_value="Ответ")
        chunk = self._context("Начало " + "Т" * 500 + " ХВОСТ")

        with patch("agents.llm._chat_completion", new=completion):
            await generate_answer(
                "вопрос",
                [chunk],
                [],
                PremiseCheck(),
                "full",
                [],
                settings=make_settings(max_answer_context_chars=100),
            )

        messages = completion.await_args.kwargs["messages"]
        self.assertNotIn("ХВОСТ", messages[1].content)


class ChatCompletionTest(unittest.IsolatedAsyncioTestCase):
    async def test_missing_settings_raise_value_error(self) -> None:
        with self.assertRaises(ValueError) as raised:
            await _chat_completion(settings=Settings(), messages=[])

        self.assertIn("LLM_BASE_URL", str(raised.exception))
        self.assertIn("LLM_API_KEY", str(raised.exception))
        self.assertIn("LLM_MODEL", str(raised.exception))

    def test_settings_repr_hides_api_key(self) -> None:
        settings = make_settings()
        self.assertNotIn("test-key", repr(settings))


class ChatCompletionLoggingTest(unittest.IsolatedAsyncioTestCase):
    async def test_logs_prompt_and_raw_response(self) -> None:
        fake_llm = SimpleNamespace(
            ainvoke=AsyncMock(return_value=SimpleNamespace(content="сырой ответ"))
        )
        with (
            patch("agents.llm._build_llm", return_value=fake_llm),
            self.assertLogs("agents.llm", level="INFO") as captured,
        ):
            result = await _chat_completion(
                settings=make_settings(),
                messages=[SystemMessage(content="системный промпт")],
                label="test",
            )

        self.assertEqual(result, "сырой ответ")
        logs = "\n".join(captured.output)
        self.assertIn("[test] LLM запрос", logs)
        self.assertIn("системный промпт", logs)
        self.assertIn("сырой ответ", logs)
        self.assertIn("LLM вызов завершён", logs)

    async def test_api_error_is_soft_and_logged(self) -> None:
        fake_llm = SimpleNamespace(
            ainvoke=AsyncMock(side_effect=RuntimeError("сеть недоступна"))
        )
        with (
            patch("agents.llm._build_llm", return_value=fake_llm),
            self.assertLogs("agents.llm", level="WARNING") as captured,
        ):
            result = await _chat_completion(
                settings=make_settings(),
                messages=[SystemMessage(content="промпт")],
            )

        self.assertEqual(result, "")
        self.assertIn("RAG-fallback", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
