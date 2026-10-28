"""Own-provider requests must never use the app's client or shared breaker."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm import llm_router as router


class OwnCredentialsTest(unittest.TestCase):
    def tearDown(self):
        router.set_user_credentials(None)

    def test_cloudflare_rewrite_and_answer_use_own_client(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            if kwargs.get("stream"):
                return [SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=123))], usage=None)]
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="rewrite"), finish_reason="stop")], usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        llm = router.LLMRouter.__new__(router.LLMRouter)
        setattr(llm, "_cf", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: (_ for _ in ()).throw(AssertionError("app charged"))))))
        llm._gemini = None
        llm.last_stream_usage = {}
        router.set_user_credentials("CLOUDFLARE", "test-token", "a" * 32)
        with patch.object(router, "_user_cloudflare_client", return_value=client) as build:
            self.assertEqual(llm.complete("rewrite").text, "rewrite")
            self.assertEqual("".join(llm.stream("answer")), "123")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["model"], router.CLOUDFLARE_MODEL)
        build.assert_called_with("a" * 32, "test-token")

    def test_cloudflare_failure_never_uses_app_credentials(self):
        llm = router.LLMRouter.__new__(router.LLMRouter)
        llm._cf = None
        llm._gemini = None
        llm.last_stream_usage = {}
        router.set_user_credentials("CLOUDFLARE", "bad-token", "a" * 32)
        with patch.object(router, "_user_cloudflare_client", side_effect=ValueError("bad token")), \
             patch.object(llm, "_client_for", side_effect=AssertionError("app charged")):
            self.assertEqual(llm.complete("question").text, "")
            self.assertEqual(list(llm.stream("question")), [])

    def test_gemini_key_check_keeps_client_open_for_request(self):
        client = MagicMock()
        client.__enter__.return_value = client
        with patch.object(router, "_user_client", return_value=client):
            router.check_user_key("test-key")
        client.models.get.assert_called_once_with(model=router.GEMINI_CHAT_MODEL)
        client.__enter__.assert_called_once_with()
        client.__exit__.assert_called_once()

    def test_gemini_generation_settings_match_model_family(self):
        modern = router._gemini_config("gemini-3.5-flash", 0.3, 8192, 2048)
        self.assertEqual(modern.thinking_config.thinking_level, router.types.ThinkingLevel.MEDIUM)
        self.assertIsNone(modern.thinking_config.thinking_budget)
        self.assertIsNone(modern.temperature)
        self.assertEqual(modern.max_output_tokens, 8192)
        minimal = router._gemini_config("gemini-3.5-flash", 0, 512, 0)
        self.assertEqual(minimal.thinking_config.thinking_level, router.types.ThinkingLevel.MINIMAL)
        legacy = router._gemini_config("gemini-2.5-flash-lite", 0, 4096, 0)
        self.assertEqual(legacy.thinking_config.thinking_budget, 0)
        self.assertIsNone(legacy.thinking_config.thinking_level)
        self.assertEqual(legacy.temperature, 0)

    def test_gemini_stream_and_completion_use_new_model_settings(self):
        client = MagicMock()
        client.models.generate_content.return_value = SimpleNamespace(text="OK", usage_metadata=None)
        client.models.generate_content_stream.return_value = [SimpleNamespace(text="OK", usage_metadata=None)]
        llm = router.LLMRouter.__new__(router.LLMRouter)
        llm._cf = None
        llm._gemini = None
        router.set_user_credentials("GEMINI", "test-key")
        with patch.object(router, "_user_client", return_value=client):
            self.assertEqual(llm.complete("question").text, "OK")
            self.assertEqual("".join(llm.stream("question")), "OK")
        for call in (client.models.generate_content.call_args, client.models.generate_content_stream.call_args):
            self.assertEqual(call.kwargs["model"], router.GEMINI_CHAT_MODEL)
            self.assertEqual(call.kwargs["config"].thinking_config.thinking_level, router.types.ThinkingLevel.MEDIUM)

    def test_failed_own_key_does_not_fall_back_to_app(self):
        llm = router.LLMRouter.__new__(router.LLMRouter)
        llm._cf = None
        llm._gemini = None
        router.set_user_credentials("GEMINI", "invalid-key")
        with patch.object(router, "_user_client", side_effect=ValueError("bad key")), \
             patch.object(llm, "_client_for", side_effect=AssertionError("app charged")):
            self.assertEqual(llm.complete("question").text, "")
        client = SimpleNamespace(models=SimpleNamespace(
            generate_content_stream=lambda **kwargs: (_ for _ in ()).throw(ValueError("bad key"))))
        with patch.object(router, "_user_client", return_value=client):
            self.assertEqual(list(llm.stream("question")), [])


if __name__ == "__main__":
    unittest.main()
