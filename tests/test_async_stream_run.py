"""
Tests for async streaming in _run_internal.

Regression test for bug where AsyncPromptLayer._run_internal called
response.model_dump() before checking if stream=True, causing
AttributeError on AsyncStream objects.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptlayer import AsyncPromptLayer


@pytest.fixture
def mock_template():
    return {
        "id": 1,
        "prompt_template": {"type": "chat", "messages": []},
        "metadata": {"model": {"provider": "openai", "name": "gpt-4o", "parameters": {}}},
        "llm_kwargs": {"model": "gpt-4o"},
    }


@pytest.fixture
def mock_llm_data():
    return {
        "request_function": None,
        "provider": "openai",
        "function_name": "test",
        "stream_function": AsyncMock(),
        "client_kwargs": {},
        "function_kwargs": {},
        "prompt_blueprint": {"id": 1, "version": 1, "metadata": {"model": {"provider": "openai"}}},
    }


class FakeAsyncStream:
    """Mimics an LLM AsyncStream object that does NOT have model_dump."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class TestAsyncStreamRun:
    @pytest.mark.asyncio
    @patch("promptlayer.templates.AsyncTemplateManager.get")
    async def test_async_run_stream_does_not_call_model_dump(self, mock_template_get, mock_template, mock_llm_data):
        """stream=True should return early before trying to serialize the response."""
        mock_template_get.return_value = mock_template

        fake_stream = FakeAsyncStream()

        async def mock_request(**kwargs):
            return fake_stream

        client = AsyncPromptLayer(api_key="test_key")
        with patch.object(client, "_prepare_llm_data") as mock_prepare:
            mock_prepare.return_value = {**mock_llm_data, "request_function": mock_request}

            result = await client._run_internal(prompt_name="test_prompt", stream=True)

        # Should return an async generator, not crash
        assert hasattr(result, "__aiter__")

    @pytest.mark.asyncio
    @patch("promptlayer.templates.AsyncTemplateManager.get")
    async def test_async_run_stream_with_model_dump_object(self, mock_template_get, mock_template, mock_llm_data):
        """Even if the stream object has model_dump, stream=True should skip serialization."""
        mock_template_get.return_value = mock_template

        stream_with_dump = MagicMock()
        stream_with_dump.model_dump.side_effect = AttributeError("'AsyncStream' object has no attribute 'model_dump'")
        stream_with_dump.__aiter__ = MagicMock(return_value=stream_with_dump)
        stream_with_dump.__anext__ = AsyncMock(side_effect=StopAsyncIteration)

        async def mock_request(**kwargs):
            return stream_with_dump

        client = AsyncPromptLayer(api_key="test_key")
        with patch.object(client, "_prepare_llm_data") as mock_prepare:
            mock_prepare.return_value = {**mock_llm_data, "request_function": mock_request}

            # This should NOT raise AttributeError
            result = await client._run_internal(prompt_name="test_prompt", stream=True)

        assert hasattr(result, "__aiter__")
        # model_dump should never have been called
        stream_with_dump.model_dump.assert_not_called()

    @pytest.mark.asyncio
    @patch("promptlayer.promptlayer.atrack_request")
    @patch("promptlayer.templates.AsyncTemplateManager.get")
    async def test_async_run_non_stream_still_serializes(
        self, mock_template_get, mock_atrack_request, mock_template, mock_llm_data
    ):
        """stream=False should still call model_dump to serialize the response."""
        mock_template_get.return_value = mock_template

        mock_response = MagicMock()
        mock_response.model_dump = MagicMock(return_value={"choices": []})

        async def mock_request(**kwargs):
            return mock_response

        async def capture_track_request(*args, **kwargs):
            return {"request_id": "test_id", "prompt_blueprint": {}}

        mock_atrack_request.side_effect = capture_track_request

        client = AsyncPromptLayer(api_key="test_key")
        with patch.object(client, "_prepare_llm_data") as mock_prepare:
            mock_prepare.return_value = {**mock_llm_data, "request_function": mock_request}

            result = await client._run_internal(prompt_name="test_prompt", stream=False)

        mock_response.model_dump.assert_called_once_with(mode="json")
        assert result["request_id"] == "test_id"
