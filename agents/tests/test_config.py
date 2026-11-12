from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agents.config import Settings, get_settings


class SettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        get_settings.cache_clear()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(get_settings.cache_clear)
        self.addCleanup(self.env.stop)

    def test_default_limits(self) -> None:
        settings = get_settings()

        self.assertEqual(settings.max_merged_chunks, 8)
        self.assertEqual(settings.max_grade_context_chars, 4000)
        self.assertEqual(settings.max_answer_context_chars, 5000)
        self.assertEqual(settings.max_rewrite_subqueries_per_round, 2)
        self.assertEqual(settings.max_total_tokens_per_question, 20000)
        self.assertEqual(settings.max_total_tokens_soft, 15000)
        self.assertEqual(settings.history_limit, 8)

    def test_env_overrides_limits(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MAX_GRADE_CONTEXT_CHARS": "3000",
                "MAX_ANSWER_CONTEXT_CHARS": "4500",
                "MAX_REWRITE_SUBQUERIES_PER_ROUND": "1",
                "MAX_TOTAL_TOKENS_PER_QUESTION": "18000",
                "MAX_TOTAL_TOKENS_SOFT": "12000",
                "MAX_MERGED_CHUNKS": "6",
                "HISTORY_LIMIT": "6",
            },
        ):
            settings = get_settings()

        self.assertEqual(settings.max_grade_context_chars, 3000)
        self.assertEqual(settings.max_answer_context_chars, 4500)
        self.assertEqual(settings.max_rewrite_subqueries_per_round, 1)
        self.assertEqual(settings.max_total_tokens_per_question, 18000)
        self.assertEqual(settings.max_total_tokens_soft, 12000)
        self.assertEqual(settings.max_merged_chunks, 6)
        self.assertEqual(settings.history_limit, 6)

    def test_legacy_context_chars_fall_back_to_answer_context(self) -> None:
        with patch.dict(os.environ, {"MAX_CONTEXT_CHARS": "7777"}):
            settings = get_settings()

        self.assertEqual(settings.max_answer_context_chars, 7777)

    def test_answer_context_overrides_legacy_context_chars(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MAX_CONTEXT_CHARS": "7777",
                "MAX_ANSWER_CONTEXT_CHARS": "6666",
            },
        ):
            settings = get_settings()

        self.assertEqual(settings.max_answer_context_chars, 6666)

    def test_soft_budget_must_be_lower_than_hard_budget(self) -> None:
        with self.assertRaises(ValueError):
            Settings(
                max_total_tokens_per_question=15000,
                max_total_tokens_soft=15000,
            )

    def test_positive_limits_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            Settings(max_rewrite_subqueries_per_round=0)
        with self.assertRaises(ValueError):
            Settings(max_grade_context_chars=0)
        with self.assertRaises(ValueError):
            Settings(max_answer_context_chars=0)


if __name__ == "__main__":
    unittest.main()
