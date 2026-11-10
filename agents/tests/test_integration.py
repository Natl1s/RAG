from __future__ import annotations

import asyncio
import os
import unittest

from agents.bootstrap import DEFAULT_DATA_DIR

_HAS_DATA = any(DEFAULT_DATA_DIR.glob("*.parquet"))
_RUN_INTEGRATION = os.getenv("RUN_RAG_INTEGRATION") == "1"


@unittest.skipUnless(_HAS_DATA, "parquet не найден")
@unittest.skipUnless(_RUN_INTEGRATION, "установите RUN_RAG_INTEGRATION=1")
class RagIntegrationTest(unittest.TestCase):
    def test_real_rag_returns_sources(self) -> None:
        from agents.bootstrap import build_local_service

        service = build_local_service(limit=100)
        result = asyncio.run(
            service.aanswer("integration", "Какие налоги платит ИП?")
        )

        self.assertTrue(result.answer)
        self.assertTrue(result.sources)


if __name__ == "__main__":
    unittest.main()
