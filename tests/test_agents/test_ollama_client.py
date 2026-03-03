"""Tests for the shared Ollama chat client."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from researchforge.agents.ollama_client import (
    OllamaResponseError,
    OllamaTimeoutError,
    ollama_chat,
)


def _make_ollama_response(content: str, input_tokens: int = 100, output_tokens: int = 50):
    """Create a mock Ollama /api/chat response."""
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": content},
            "prompt_eval_count": input_tokens,
            "eval_count": output_tokens,
        },
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )


class TestOllamaChatBasic:
    async def test_returns_text_response(self):
        mock_resp = _make_ollama_response("Hello, world!")
        with patch("researchforge.agents.ollama_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await ollama_chat(
                model="test-model:7b",
                system_prompt="You are helpful.",
                user_message="Hi",
                agent_name="test",
            )

        assert result["content"] == "Hello, world!"
        assert result["parsed"] is None
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["fallback_used"] is False
        assert result["model"] == "test-model:7b"

    async def test_returns_json_response(self):
        json_content = json.dumps({"plan": ["step1", "step2"]})
        mock_resp = _make_ollama_response(json_content)
        with patch("researchforge.agents.ollama_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await ollama_chat(
                model="test-model:7b",
                system_prompt="Respond in JSON.",
                user_message="Plan something.",
                expect_json=True,
                agent_name="planner",
            )

        assert result["parsed"] == {"plan": ["step1", "step2"]}
        assert result["content"] == json_content


class TestOllamaChatRetry:
    async def test_retries_on_invalid_json(self):
        """First attempt returns invalid JSON, second returns valid."""
        bad_resp = _make_ollama_response("not valid json {{{")
        good_resp = _make_ollama_response(json.dumps({"result": "ok"}))

        with patch("researchforge.agents.ollama_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [bad_resp, good_resp]
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await ollama_chat(
                model="test-model:7b",
                system_prompt="JSON only.",
                user_message="Do it.",
                expect_json=True,
                agent_name="test",
            )

        assert result["parsed"] == {"result": "ok"}
        assert mock_client.post.call_count == 2

    async def test_raises_after_max_retries(self):
        """All attempts return invalid JSON → OllamaResponseError."""
        bad_resp = _make_ollama_response("not json")

        with patch("researchforge.agents.ollama_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = bad_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(OllamaResponseError, match="Failed to get valid JSON"):
                await ollama_chat(
                    model="test-model:7b",
                    system_prompt="JSON only.",
                    user_message="Do it.",
                    expect_json=True,
                    agent_name="test",
                )

        # 1 initial + 2 retries = 3
        assert mock_client.post.call_count == 3


class TestOllamaChatFallback:
    async def test_falls_back_on_timeout(self):
        """Primary model times out, fallback model succeeds."""
        good_resp = _make_ollama_response("fallback response")

        with (
            patch("researchforge.agents.ollama_client.httpx.AsyncClient") as mock_cls,
            patch(
                "researchforge.agents.ollama_client._model_sequence",
                return_value=["primary:14b", "fallback:7b"],
            ),
        ):
            mock_client = AsyncMock()
            # First call (primary) times out, second (fallback) succeeds
            mock_client.post.side_effect = [
                httpx.TimeoutException("timeout"),
                good_resp,
            ]
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await ollama_chat(
                model="primary:14b",
                system_prompt="test",
                user_message="test",
                agent_name="analyst",
            )

        assert result["content"] == "fallback response"
        assert result["fallback_used"] is True
        assert result["model"] == "fallback:7b"

    async def test_raises_if_all_models_timeout(self):
        """Both primary and fallback time out."""
        with (
            patch("researchforge.agents.ollama_client.httpx.AsyncClient") as mock_cls,
            patch(
                "researchforge.agents.ollama_client._model_sequence",
                return_value=["primary:14b", "fallback:7b"],
            ),
        ):
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("timeout")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(OllamaTimeoutError, match="All models timed out"):
                await ollama_chat(
                    model="primary:14b",
                    system_prompt="test",
                    user_message="test",
                    agent_name="analyst",
                )


class TestModelSequence:
    def test_returns_primary_only_when_no_fallback(self):
        from researchforge.agents.ollama_client import _model_sequence

        # "writer" has a fallback in default config, but "researcher" doesn't
        # Use an agent name not in fallbacks
        result = _model_sequence("some-model:7b", "gatherer")
        assert result == ["some-model:7b"]

    def test_returns_primary_and_fallback(self):
        from researchforge.agents.ollama_client import _model_sequence

        # "planner" has a fallback of "qwen2.5:7b" in default config
        result = _model_sequence("deepseek-r1:14b", "planner")
        assert len(result) == 2
        assert result[0] == "deepseek-r1:14b"
        assert result[1] == "qwen2.5:7b"
