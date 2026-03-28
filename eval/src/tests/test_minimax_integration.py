"""Integration tests for MiniMaxModel with live API calls.

These tests require:
  - MINIMAX_API_KEY environment variable to be set
  - Network access to https://api.minimax.io/v1

Run with: python -m pytest eval/src/tests/test_minimax_integration.py -v
Skip if no API key: tests are automatically skipped when MINIMAX_API_KEY is not set.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
SKIP_REASON = "MINIMAX_API_KEY not set; skipping integration tests"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxModelIntegration(unittest.TestCase):
    """Live integration tests against MiniMax API."""

    def test_achat_returns_response(self):
        from models import MiniMaxModel

        model = MiniMaxModel(ver="MiniMax-M2.7")
        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(
                model.achat("什么是合同？请用一句话回答。")
            )
        finally:
            loop.close()
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_returns_response(self):
        from models import MiniMaxModel

        model = MiniMaxModel(ver="MiniMax-M2.7")
        response = model.chat("1+1等于几？请只回答数字。")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_mcq_format_and_chat(self):
        """Test that MiniMax can answer a formatted MCQ question."""
        from models import MiniMaxModel

        model = MiniMaxModel(ver="MiniMax-M2.7")
        query = (
            "以下是一道单项选择题，不需要做任何分析和解释，直接输出答案选项。\n\n"
            "问题：根据我国《民法典》，自然人的民事权利能力始于何时？\n"
            "A. 受孕时\n"
            "B. 出生时\n"
            "C. 年满十八周岁时\n"
            "D. 取得身份证时\n"
            "答案："
        )
        response = model.chat(query)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)


if __name__ == "__main__":
    unittest.main()
