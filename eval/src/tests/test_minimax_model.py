"""Unit tests for MiniMaxModel in the DISC-LawLLM evaluation framework."""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path so we can import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import BaseModel, MiniMaxModel, MODELS, get_model


class TestMiniMaxModelRegistration(unittest.TestCase):
    """Test that MiniMax models are properly registered in the MODELS dict."""

    def test_minimax_m27_registered(self):
        self.assertIn("minimax-m2.7", MODELS)
        self.assertEqual(MODELS["minimax-m2.7"]["klass"], MiniMaxModel)
        self.assertEqual(MODELS["minimax-m2.7"]["kwargs"]["ver"], "MiniMax-M2.7")

    def test_minimax_m27_highspeed_registered(self):
        self.assertIn("minimax-m2.7-highspeed", MODELS)
        self.assertEqual(MODELS["minimax-m2.7-highspeed"]["kwargs"]["ver"], "MiniMax-M2.7-highspeed")

    def test_minimax_m25_registered(self):
        self.assertIn("minimax-m2.5", MODELS)
        self.assertEqual(MODELS["minimax-m2.5"]["kwargs"]["ver"], "MiniMax-M2.5")

    def test_minimax_m25_highspeed_registered(self):
        self.assertIn("minimax-m2.5-highspeed", MODELS)
        self.assertEqual(MODELS["minimax-m2.5-highspeed"]["kwargs"]["ver"], "MiniMax-M2.5-highspeed")

    def test_all_minimax_use_correct_class(self):
        for key in MODELS:
            if key.startswith("minimax"):
                self.assertEqual(MODELS[key]["klass"], MiniMaxModel)


class TestMiniMaxModelInit(unittest.TestCase):
    """Test MiniMaxModel initialization."""

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MINIMAX_API_KEY", None)
            with self.assertRaises(ValueError) as ctx:
                MiniMaxModel(ver="MiniMax-M2.7")
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    def test_invalid_model_raises(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with self.assertRaises(ValueError) as ctx:
                MiniMaxModel(ver="invalid-model")
            self.assertIn("invalid-model", str(ctx.exception))

    def test_valid_init(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            model = MiniMaxModel(ver="MiniMax-M2.7")
            self.assertEqual(model.ver, "MiniMax-M2.7")
            self.assertIsNotNone(model.client)

    def test_default_model_is_m27(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            model = MiniMaxModel()
            self.assertEqual(model.ver, "MiniMax-M2.7")

    def test_all_supported_models_init(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            for ver in MiniMaxModel.SUPPORTED_MODELS:
                model = MiniMaxModel(ver=ver)
                self.assertEqual(model.ver, ver)

    def test_client_base_url(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            model = MiniMaxModel(ver="MiniMax-M2.7")
            self.assertEqual(str(model.client.base_url), "https://api.minimax.io/v1/")


class TestMiniMaxModelBaseClass(unittest.TestCase):
    """Test MiniMaxModel inherits from BaseModel."""

    def test_inherits_base_model(self):
        self.assertTrue(issubclass(MiniMaxModel, BaseModel))

    def test_has_chat_method(self):
        self.assertTrue(hasattr(MiniMaxModel, "chat"))

    def test_has_achat_method(self):
        self.assertTrue(hasattr(MiniMaxModel, "achat"))

    def test_has_mcq_formatter(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            model = MiniMaxModel()
            self.assertTrue(hasattr(model, "mcq_formatter"))

    def test_has_qa_formatter(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            model = MiniMaxModel()
            self.assertTrue(hasattr(model, "qa_formatter"))


class TestMiniMaxModelFormatMessage(unittest.TestCase):
    """Test message formatting."""

    def setUp(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            self.model = MiniMaxModel(ver="MiniMax-M2.7")

    def test_format_message_structure(self):
        messages = self.model._format_message("什么是合同法？")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    def test_format_message_system_prompt(self):
        messages = self.model._format_message("test")
        self.assertIn("法律专家", messages[0]["content"])

    def test_format_message_user_content(self):
        query = "什么是民事诉讼？"
        messages = self.model._format_message(query)
        self.assertEqual(messages[1]["content"], query)


class TestMiniMaxModelProcessCompletion(unittest.TestCase):
    """Test completion processing including think-tag stripping."""

    def setUp(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            self.model = MiniMaxModel(ver="MiniMax-M2.5")

    def _make_completion(self, content):
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = content
        return completion

    def test_normal_response(self):
        completion = self._make_completion("这是一个法律回答。")
        result = self.model._process_completion(completion)
        self.assertEqual(result, "这是一个法律回答。")

    def test_strips_think_tags(self):
        completion = self._make_completion(
            "<think>让我分析一下这个法律问题...</think>\n答案是A。"
        )
        result = self.model._process_completion(completion)
        self.assertEqual(result, "答案是A。")

    def test_strips_multiline_think_tags(self):
        completion = self._make_completion(
            "<think>\n这是一个复杂的问题\n需要仔细分析\n</think>\n\n根据合同法第42条..."
        )
        result = self.model._process_completion(completion)
        self.assertEqual(result, "根据合同法第42条...")

    def test_no_think_tags_unmodified(self):
        text = "根据《刑法》第23条，犯罪未遂是指..."
        completion = self._make_completion(text)
        result = self.model._process_completion(completion)
        self.assertEqual(result, text)

    def test_empty_response(self):
        completion = self._make_completion("")
        result = self.model._process_completion(completion)
        self.assertEqual(result, "")


class TestMiniMaxModelChat(unittest.TestCase):
    """Test the chat and achat methods."""

    def setUp(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            self.model = MiniMaxModel(ver="MiniMax-M2.7")

    def test_achat_calls_api(self):
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "答案是B"

        async def mock_create(**kwargs):
            return mock_completion

        self.model.client = MagicMock()
        self.model.client.chat.completions.create = mock_create

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.model.achat("什么是侵权？"))
        finally:
            loop.close()
        self.assertEqual(result, "答案是B")

    def test_achat_uses_correct_model(self):
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "response"

        captured_kwargs = {}

        async def mock_create(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_completion

        self.model.client = MagicMock()
        self.model.client.chat.completions.create = mock_create

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.model.achat("test"))
        finally:
            loop.close()

        self.assertEqual(captured_kwargs["model"], "MiniMax-M2.7")

    def test_achat_uses_temperature(self):
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "response"

        captured_kwargs = {}

        async def mock_create(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_completion

        self.model.client = MagicMock()
        self.model.client.chat.completions.create = mock_create

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.model.achat("test"))
        finally:
            loop.close()

        self.assertEqual(captured_kwargs["temperature"], 0.1)

    def test_chat_delegates_to_achat(self):
        with patch.object(self.model, "achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = "test response"
            result = self.model.chat("test query")
            mock_achat.assert_called_once_with("test query")
            self.assertEqual(result, "test response")


class TestGetModelMiniMax(unittest.TestCase):
    """Test get_model() factory function with MiniMax models."""

    def test_get_model_minimax_m27(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            model = get_model("minimax-m2.7")
            self.assertIsInstance(model, MiniMaxModel)
            self.assertEqual(model.ver, "MiniMax-M2.7")

    def test_get_model_minimax_m25(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            model = get_model("minimax-m2.5")
            self.assertIsInstance(model, MiniMaxModel)
            self.assertEqual(model.ver, "MiniMax-M2.5")

    def test_get_model_eval_only_returns_null(self):
        model = get_model("minimax-m2.7", eval_only=True)
        from models import NullModel
        self.assertIsInstance(model, NullModel)


class TestMiniMaxModelSupportedModels(unittest.TestCase):
    """Test the SUPPORTED_MODELS constant."""

    def test_supported_models_is_list(self):
        self.assertIsInstance(MiniMaxModel.SUPPORTED_MODELS, list)

    def test_supported_models_not_empty(self):
        self.assertTrue(len(MiniMaxModel.SUPPORTED_MODELS) > 0)

    def test_m27_in_supported(self):
        self.assertIn("MiniMax-M2.7", MiniMaxModel.SUPPORTED_MODELS)

    def test_m25_in_supported(self):
        self.assertIn("MiniMax-M2.5", MiniMaxModel.SUPPORTED_MODELS)

    def test_highspeed_variants_in_supported(self):
        self.assertIn("MiniMax-M2.7-highspeed", MiniMaxModel.SUPPORTED_MODELS)
        self.assertIn("MiniMax-M2.5-highspeed", MiniMaxModel.SUPPORTED_MODELS)


if __name__ == "__main__":
    unittest.main()
