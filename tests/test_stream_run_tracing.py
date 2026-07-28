"""Regression: stream=True must not record the generator as function_output."""

from unittest.mock import MagicMock, patch

import pytest

from promptlayer import AsyncPromptLayer, PromptLayer
from promptlayer.tracing_context import format_run_output, is_stream_result


def test_is_stream_result_detects_generators():
    def sync_gen():
        yield 1

    async def async_gen():
        yield 1

    assert is_stream_result(sync_gen())
    assert is_stream_result(async_gen())
    assert not is_stream_result({"request_id": "1"})
    assert not is_stream_result("text")


def test_format_run_output_prefers_prompt_blueprint():
    assert format_run_output({"prompt_blueprint": {"messages": []}, "raw_response": "x"}) == str({"messages": []})
    assert format_run_output({"raw_response": "hello"}) == "hello"


@patch("promptlayer.templates.TemplateManager.get")
def test_sync_run_stream_sets_function_output_after_consume(mock_template_get):
    mock_template_get.return_value = {
        "id": 1,
        "prompt_template": {"type": "chat", "messages": []},
        "metadata": {"model": {"provider": "openai", "name": "gpt-4o", "parameters": {}}},
        "llm_kwargs": {"model": "gpt-4o"},
    }

    final_chunk = {
        "request_id": "req_1",
        "raw_response": "chunk",
        "prompt_blueprint": {"messages": [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]},
    }

    def fake_stream():
        yield {"request_id": None, "raw_response": "partial", "prompt_blueprint": None}
        yield final_chunk

    client = PromptLayer(api_key="test_key", enable_tracing=True)
    span = MagicMock()
    span.get_span_context.return_value = MagicMock(span_id=1, is_valid=True)
    client.tracer = MagicMock()
    client.tracer.start_span.return_value = span

    with patch.object(client, "_run_internal", return_value=fake_stream()):
        stream = client.run(prompt_name="test_prompt", stream=True)
        # Before consume, function_output must not be set to the generator.
        span.set_attribute.assert_any_call("prompt_name", "test_prompt")
        output_calls = [c for c in span.set_attribute.call_args_list if c[0][0] == "function_output"]
        assert output_calls == []

        chunks = list(stream)

    assert len(chunks) == 2
    output_calls = [c for c in span.set_attribute.call_args_list if c[0][0] == "function_output"]
    assert len(output_calls) == 1
    assert "<generator" not in output_calls[0][0][1]
    assert "hi" in output_calls[0][0][1]
    span.end.assert_called_once()


@pytest.mark.asyncio
@patch("promptlayer.templates.AsyncTemplateManager.get")
async def test_async_run_stream_sets_function_output_after_consume(mock_template_get):
    mock_template_get.return_value = {
        "id": 1,
        "prompt_template": {"type": "chat", "messages": []},
        "metadata": {"model": {"provider": "openai", "name": "gpt-4o", "parameters": {}}},
        "llm_kwargs": {"model": "gpt-4o"},
    }

    final_chunk = {
        "request_id": "req_1",
        "raw_response": "chunk",
        "prompt_blueprint": {"messages": [{"role": "assistant", "content": [{"type": "text", "text": "bye"}]}]},
    }

    async def fake_stream():
        yield {"request_id": None, "raw_response": "partial", "prompt_blueprint": None}
        yield final_chunk

    client = AsyncPromptLayer(api_key="test_key", enable_tracing=True)
    span = MagicMock()
    span.get_span_context.return_value = MagicMock(span_id=1, is_valid=True)
    client.tracer = MagicMock()
    client.tracer.start_span.return_value = span

    async def mock_run_internal(**kwargs):
        return fake_stream()

    with patch.object(client, "_run_internal", side_effect=mock_run_internal):
        stream = await client.run(prompt_name="test_prompt", stream=True)
        output_calls = [c for c in span.set_attribute.call_args_list if c[0][0] == "function_output"]
        assert output_calls == []

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

    assert len(chunks) == 2
    output_calls = [c for c in span.set_attribute.call_args_list if c[0][0] == "function_output"]
    assert len(output_calls) == 1
    assert "<async_generator" not in output_calls[0][0][1]
    assert "bye" in output_calls[0][0][1]
    span.end.assert_called_once()
