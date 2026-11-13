from __future__ import annotations

import unittest

from agents.clarification import (
    candidate_documents,
    detect_context_ambiguity,
    last_assistant_asked_question,
)
from agents.state import ContextChunk, DialogTurn

ALFA_TRADE = ContextChunk(
    doc_id="документы компании__Договор поставки П-17-2024 Альфа-Трейд.txt",
    chunk_id="alfa-1",
    text="Гарантийный срок — 24 месяца.",
    metadata={
        "title": "документы компании__Договор поставки П-17-2024 Альфа-Трейд.txt"
    },
    score=0.9,
    subquery_ids=[1],
)

ALFA_TRADING = ContextChunk(
    doc_id="документы компании__Договор поставки ПТ-05-2023 Альфа-Трейдинг.txt",
    chunk_id="trading-1",
    text="Гарантийный срок — 12 месяцев.",
    metadata={
        "title": "документы компании__Договор поставки ПТ-05-2023 Альфа-Трейдинг.txt"
    },
    score=0.8,
    subquery_ids=[1],
)

SEVERNY_SKLAD = ContextChunk(
    doc_id="документы компании__Договор аренды А-08-2024 Северный Склад.txt",
    chunk_id="sklad-1",
    text="Арендная плата — 450 000 рублей.",
    metadata={
        "title": "документы компании__Договор аренды А-08-2024 Северный Склад.txt"
    },
    score=0.95,
    subquery_ids=[1],
)

TAX_LAW = ContextChunk(
    doc_id="законы__НК_ст112_действующая.txt",
    chunk_id="tax-1",
    text="Налоговые санкции.",
    metadata={"title": "законы__НК_ст112_действующая.txt"},
    score=0.7,
    subquery_ids=[1],
)


class CandidateDocumentsTest(unittest.TestCase):
    def test_extracts_numbers_and_names(self) -> None:
        candidates = candidate_documents([ALFA_TRADE, ALFA_TRADING, TAX_LAW])

        self.assertEqual(len(candidates), 2)
        labels = {candidate.label for candidate in candidates}
        self.assertIn("Альфа-Трейд (П-17-2024)", labels)
        self.assertIn("Альфа-Трейдинг (ПТ-05-2023)", labels)

    def test_prefers_title_over_long_header_for_name(self) -> None:
        chunk = ContextChunk(
            doc_id="документы компании__Договор поставки П-17-2024 Альфа-Трейд.txt",
            chunk_id="alfa-1",
            text="Текст",
            metadata={
                "title": "документы компании__Договор поставки П-17-2024 Альфа-Трейд.txt",
                "head_doc": (
                    "ДОГОВОР ПОСТАВКИ № П-17/2024 г. Москва — 12 февраля 2024 г. "
                    "ООО «Альфа-Трейд»"
                ),
            },
            score=0.9,
            subquery_ids=[1],
        )

        candidates = candidate_documents([chunk])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].label, "Альфа-Трейд (П-17/2024)")

    def test_falls_back_to_header_name_without_title(self) -> None:
        chunk = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Текст",
            metadata={"head_doc": "ДОГОВОР ПОСТАВКИ № П-17/2024 Альфа-Трейд"},
            score=0.9,
            subquery_ids=[1],
        )

        candidates = candidate_documents([chunk])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].label, "Альфа-Трейд (П-17/2024)")

    def test_ignores_non_contract_documents_with_numbers(self) -> None:
        letter = ContextChunk(
            doc_id="документы компании__Письмо комплаенс-службы К-12 от 15.04.2025.txt",
            chunk_id="letter-1",
            text="Текст",
            metadata={
                "head_doc": (
                    "ООО «МЕРИДИАН» Служба комплаенс Исх. № К-12 от 15.04.2025"
                )
            },
            score=0.9,
            subquery_ids=[1],
        )
        regulation = ContextChunk(
            doc_id="документы компании__Регламент согласования договоров ред 2025.txt",
            chunk_id="reg-1",
            text="Текст",
            metadata={
                "head_doc": (
                    "РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ ООО «МЕРИДИАН» "
                    "Утверждён приказом генерального директора № 14-П"
                )
            },
            score=0.8,
            subquery_ids=[1],
        )

        candidates = candidate_documents([letter, regulation])

        self.assertEqual(candidates, [])

    def test_deduplicates_same_contract(self) -> None:
        supplement = ContextChunk(
            doc_id="документы компании__Доп соглашение 1 к договору П-17-2024.txt",
            chunk_id="supplement-1",
            text="Дополнительное соглашение.",
            metadata={
                "title": "документы компании__Доп соглашение 1 к договору П-17-2024.txt"
            },
            score=0.7,
            subquery_ids=[1],
        )

        candidates = candidate_documents([ALFA_TRADE, supplement])

        self.assertEqual(len(candidates), 1)


class LastAssistantQuestionTest(unittest.TestCase):
    def test_detects_question(self) -> None:
        history = [
            DialogTurn(role="user", text="Мы съезжаем."),
            DialogTurn(role="assistant", text="Уточните номер договора?"),
        ]

        self.assertTrue(last_assistant_asked_question(history))

    def test_ignores_answer(self) -> None:
        history = [
            DialogTurn(role="user", text="Вопрос."),
            DialogTurn(role="assistant", text="Ответ по документу."),
        ]

        self.assertFalse(last_assistant_asked_question(history))


class DetectContextAmbiguityTest(unittest.TestCase):
    def test_asks_to_choose_between_companies(self) -> None:
        result = detect_context_ambiguity(
            "Подскажите гарантию по договору с «Альфой».",
            [],
            [ALFA_TRADE, ALFA_TRADING],
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.needed)
        self.assertIn("Альфа-Трейд", result.question)
        self.assertIn("Альфа-Трейдинг", result.question)
        self.assertEqual(result.source, "retrieval")

    def test_does_not_ask_when_company_is_explicit(self) -> None:
        result = detect_context_ambiguity(
            "Подскажите гарантию по договору с «Альфа-Трейдинг».",
            [],
            [ALFA_TRADE, ALFA_TRADING],
        )

        self.assertIsNone(result)

    def test_does_not_ask_when_company_is_in_history(self) -> None:
        history = [
            DialogTurn(role="assistant", text="Уточните компанию."),
            DialogTurn(role="user", text="«Альфа-Трейдинг», у них покупали технику."),
        ]

        result = detect_context_ambiguity(
            "На сколько гарантия?",
            history,
            [ALFA_TRADE, ALFA_TRADING],
        )

        self.assertIsNone(result)

    def test_does_not_ask_when_document_is_explicit(self) -> None:
        result = detect_context_ambiguity(
            "Что сказано в договоре П-17-2024 про гарантию?",
            [],
            [ALFA_TRADE, ALFA_TRADING],
        )

        self.assertIsNone(result)

    def test_does_not_ask_for_single_contract(self) -> None:
        result = detect_context_ambiguity(
            "Какая арендная плата по договору?",
            [],
            [SEVERNY_SKLAD],
        )

        self.assertIsNone(result)

    def test_does_not_ask_for_general_legal_question(self) -> None:
        result = detect_context_ambiguity(
            "Какой максимальный срок испытания можно установить?",
            [],
            [ALFA_TRADE, ALFA_TRADING],
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
