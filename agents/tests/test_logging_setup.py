from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.logging_setup import setup_logging


def _reset_agents_logger() -> None:
    logger = logging.getLogger("agents")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


class SetupLoggingTest(unittest.TestCase):
    def setUp(self) -> None:
        _reset_agents_logger()
        self.addCleanup(_reset_agents_logger)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log_path = Path(tmp.name) / "agent.log"

    def test_writes_file_with_run_header(self) -> None:
        env = {"LLM_LOG_FILE": str(self.log_path), "LOG_LEVEL": "INFO"}
        with patch.dict(os.environ, env):
            returned = setup_logging()

        self.assertEqual(returned, self.log_path)
        self.assertTrue(self.log_path.exists())

        logging.getLogger("agents.demo").info("сообщение из теста")

        content = self.log_path.read_text(encoding="utf-8")
        self.assertIn("=== Запуск:", content)
        self.assertIn("Логи этого запуска", content)
        self.assertIn("сообщение из теста", content)

    def test_default_unset_creates_file_in_logs_dir(self) -> None:
        env = {"LLM_LOG_FILE": "", "LOG_LEVEL": "INFO"}
        with patch.dict(os.environ, env):
            with patch("agents.logging_setup._REPO_ROOT", Path(self.log_path.parent)):
                returned = setup_logging()

        self.assertIsNotNone(returned)
        self.assertEqual(returned.parent.name, "logs")
        self.assertTrue(returned.name.startswith("llm_"))
        self.assertTrue(returned.exists())
        self.assertIn("=== Запуск:", returned.read_text(encoding="utf-8"))

    def test_off_value_disables_file(self) -> None:
        env = {"LLM_LOG_FILE": "off", "LOG_LEVEL": "INFO"}
        with patch.dict(os.environ, env):
            returned = setup_logging()

        self.assertIsNone(returned)
        handlers = logging.getLogger("agents").handlers
        self.assertNotIn(
            logging.FileHandler,
            [type(handler) for handler in handlers],
        )

    def test_second_call_is_noop(self) -> None:
        env = {"LLM_LOG_FILE": str(self.log_path), "LOG_LEVEL": "INFO"}
        with patch.dict(os.environ, env):
            first = setup_logging()
            second = setup_logging()

        self.assertEqual(first, self.log_path)
        self.assertIsNone(second)
        self.assertEqual(len(logging.getLogger("agents").handlers), 2)


if __name__ == "__main__":
    unittest.main()
