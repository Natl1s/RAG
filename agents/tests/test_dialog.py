from __future__ import annotations

import unittest

from agents.dialog import (
    format_history,
    history_records,
    parse_dialog,
    queries_similar,
)
from agents.state import DialogTurn


class ParseDialogTest(unittest.TestCase):
    def test_splits_history_and_last_user_question(self) -> None:
        raw = (
            "Пользователь: У меня есть договор № 1.\n"
            "Ассистент: Понял, чем помочь?\n"
            "Пользователь: Какие налоги по нему?"
        )

        history, last_question = parse_dialog(raw)

        self.assertEqual(
            history,
            [
                DialogTurn(role="user", text="У меня есть договор № 1."),
                DialogTurn(role="assistant", text="Понял, чем помочь?"),
            ],
        )
        self.assertEqual(last_question, "Какие налоги по нему?")

    def test_without_markers_returns_whole_text_as_question(self) -> None:
        history, last_question = parse_dialog("Какие налоги платит ИП?")

        self.assertEqual(history, [])
        self.assertEqual(last_question, "Какие налоги платит ИП?")

    def test_empty_input(self) -> None:
        self.assertEqual(parse_dialog("   "), ([], ""))

    def test_without_user_turn_returns_whole_text(self) -> None:
        history, last_question = parse_dialog("Ассистент: здравствуйте")

        self.assertEqual(history, [])
        self.assertEqual(last_question, "Ассистент: здравствуйте")


class FormatHistoryTest(unittest.TestCase):
    def test_formats_and_limits_history(self) -> None:
        history = [
            DialogTurn(role="user", text="первый"),
            DialogTurn(role="assistant", text="ответ"),
            DialogTurn(role="user", text="второй"),
        ]

        formatted = format_history(history, limit=2)

        self.assertNotIn("первый", formatted)
        self.assertIn("Ассистент: ответ", formatted)
        self.assertIn("Пользователь: второй", formatted)

    def test_history_records_limits_and_keeps_roles(self) -> None:
        history = [
            DialogTurn(role="user", text="первый"),
            DialogTurn(role="assistant", text="ответ"),
            DialogTurn(role="user", text="второй"),
            DialogTurn(role="system", text="третий"),
        ]

        records = history_records(history, limit=2)

        self.assertEqual(
            records,
            [
                {"role": "user", "text": "второй"},
                {"role": "system", "text": "третий"},
            ],
        )


class QueriesSimilarTest(unittest.TestCase):
    def test_detects_near_duplicates(self) -> None:
        self.assertTrue(queries_similar("налоги ИП", "налоги  ИП"))
        self.assertFalse(queries_similar("налоги ИП", "штрафы ИП"))


if __name__ == "__main__":
    unittest.main()
