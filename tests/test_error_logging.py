"""
Tests for error logging in PromptLayer.run() and log_request().
When an LLM call fails, the error should be tracked before re-raising.
"""

from unittest.mock import MagicMock, patch

import pytest

from promptlayer import AsyncPromptLayer, PromptLayer


@pytest.fixture
def mock_template():
    return {
        "id": 1,
        "prompt_template": {"type": "chat", "messages": []},
        "metadata": {"model": {"provider": "openai", "name": "gpt-4"}},
        "llm_kwargs": {"model": "gpt-4"},
    }


@pytest.fixture
def mock_llm_data():
    return {
        "request_function": None,  # replaced per-test
        "provider": "openai",
        "function_name": "test",
        "stream_function": None,
        "client_kwargs": {},
        "function_kwargs": {},
        "prompt_blueprint": {"id": 1, "version": 1, "metadata": {}},
    }


# ---------- Sync _run_internal ----------


class TestSyncRunInternalErrorLogging:
    @patch("promptlayer.promptlayer.track_request")
    @patch("promptlayer.templates.TemplateManager.get")
    def test_run_internal_logs_error_on_llm_failure(
        self, mock_template_get, mock_track_request, mock_template, mock_llm_data
    ):
        mock_template_get.return_value = mock_template

        llm_error = RuntimeError("model overloaded")

        def failing_request(**kwargs):
            raise llm_error

        tracked_kwargs = {}

        def capture_track_request(*args, **kwargs):
            tracked_kwargs.update(kwargs)
            return {"request_id": "test_id", "prompt_blueprint": {}}

        mock_track_request.side_effect = capture_track_request

        client = PromptLayer(api_key="test_key")
        with patch.object(client, "_prepare_llm_data") as mock_prepare:
            mock_prepare.return_value = {**mock_llm_data, "request_function": failing_request}

            with pytest.raises(RuntimeError, match="model overloaded"):
                client._run_internal(prompt_name="test_prompt", input_variables={})

        assert tracked_kwargs["status"] == "ERROR"
        assert tracked_kwargs["error_type"] == "UNKNOWN_ERROR"
        assert "model overloaded" in tracked_kwargs["error_message"]

    @patch("promptlayer.promptlayer.track_request")
    @patch("promptlayer.templates.TemplateManager.get")
    def test_run_internal_reraises_when_tracking_fails(
        self, mock_template_get, mock_track_request, mock_template, mock_llm_data
    ):
        mock_template_get.return_value = mock_template
        mock_track_request.side_effect = Exception("tracking broke")

        def failing_request(**kwargs):
            raise ValueError("original error")

        client = PromptLayer(api_key="test_key")
        with patch.object(client, "_prepare_llm_data") as mock_prepare:
            mock_prepare.return_value = {**mock_llm_data, "request_function": failing_request}

            with pytest.raises(ValueError, match="original error"):
                client._run_internal(prompt_name="test_prompt", input_variables={})

    @patch("promptlayer.promptlayer.track_request")
    @patch("promptlayer.templates.TemplateManager.get")
    def test_run_internal_no_error_tracking_on_success(
        self, mock_template_get, mock_track_request, mock_template, mock_llm_data
    ):
        mock_template_get.return_value = mock_template

        mock_response = MagicMock()
        mock_response.model_dump = MagicMock(return_value={"choices": []})

        def success_request(**kwargs):
            return mock_response

        tracked_kwargs = {}

        def capture_track_request(*args, **kwargs):
            tracked_kwargs.update(kwargs)
            return {"request_id": "test_id", "prompt_blueprint": {}}

        mock_track_request.side_effect = capture_track_request

        client = PromptLayer(api_key="test_key")
        with patch.object(client, "_prepare_llm_data") as mock_prepare:
            mock_prepare.return_value = {**mock_llm_data, "request_function": success_request}
            client._run_internal(prompt_name="test_prompt", input_variables={})

        mock_track_request.assert_called_once()
        assert "status" not in tracked_kwargs
        assert "error_type" not in tracked_kwargs
        assert "error_message" not in tracked_kwargs


# ---------- Async _run_internal ----------


class TestAsyncRunInternalErrorLogging:
    @pytest.mark.asyncio
    @patch("promptlayer.promptlayer.atrack_request")
    @patch("promptlayer.templates.AsyncTemplateManager.get")
    async def test_async_run_internal_logs_error_on_llm_failure(
        self, mock_template_get, mock_atrack_request, mock_template, mock_llm_data
    ):
        mock_template_get.return_value = mock_template

        async def failing_request(**kwargs):
            raise RuntimeError("model overloaded")

        tracked_kwargs = {}

        async def capture_track_request(*args, **kwargs):
            tracked_kwargs.update(kwargs)
            return {"request_id": "test_id", "prompt_blueprint": {}}

        mock_atrack_request.side_effect = capture_track_request

        client = AsyncPromptLayer(api_key="test_key")
        with patch.object(client, "_prepare_llm_data") as mock_prepare:
            mock_prepare.return_value = {**mock_llm_data, "request_function": failing_request}

            with pytest.raises(RuntimeError, match="model overloaded"):
                await client._run_internal(prompt_name="test_prompt", input_variables={})

        assert tracked_kwargs["status"] == "ERROR"
        assert tracked_kwargs["error_type"] == "UNKNOWN_ERROR"
        assert "model overloaded" in tracked_kwargs["error_message"]

    @pytest.mark.asyncio
    @patch("promptlayer.promptlayer.atrack_request")
    @patch("promptlayer.templates.AsyncTemplateManager.get")
    async def test_async_run_internal_reraises_when_tracking_fails(
        self, mock_template_get, mock_atrack_request, mock_template, mock_llm_data
    ):
        mock_template_get.return_value = mock_template

        async def failing_request(**kwargs):
            raise ValueError("original error")

        async def failing_track(*args, **kwargs):
            raise Exception("tracking broke")

        mock_atrack_request.side_effect = failing_track

        client = AsyncPromptLayer(api_key="test_key")
        with patch.object(client, "_prepare_llm_data") as mock_prepare:
            mock_prepare.return_value = {**mock_llm_data, "request_function": failing_request}

            with pytest.raises(ValueError, match="original error"):
                await client._run_internal(prompt_name="test_prompt", input_variables={})
