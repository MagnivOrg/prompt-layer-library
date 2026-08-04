"""Regression coverage for PromptLayer.run(stream=True) tracing."""

import asyncio
from contextvars import Context
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import context as otel_context, trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from promptlayer import AsyncPromptLayer, PromptLayer
from promptlayer.tracing_context import format_run_output, is_stream_result


def _sync_child_stream(tracer, name):
    span = tracer.start_span(name)
    token = otel_context.attach(otel_trace.set_span_in_context(span))

    def stream():
        try:
            yield {"raw_response": "chunk"}
        finally:
            otel_context.detach(token)
            span.end()

    return stream()


def _async_child_stream(tracer, name):
    span = tracer.start_span(name)
    token = otel_context.attach(otel_trace.set_span_in_context(span))

    async def stream():
        try:
            yield {"raw_response": "chunk"}
        finally:
            otel_context.detach(token)
            span.end()

    return stream()


def _assert_sequential_run_hierarchy(spans, root_span_id, current_span_ids):
    runs = [span for span in spans if span.name == "PromptLayer Run"]
    tool = next(span for span in spans if span.name == "Tool: demo")
    llm_calls = [span for span in spans if span.name.startswith("LLM call")]

    assert current_span_ids == [root_span_id, root_span_id]
    assert len(runs) == 2
    assert {span.parent.span_id for span in runs} == {root_span_id}
    assert tool.parent.span_id == root_span_id
    assert {span.parent.span_id for span in llm_calls} == {span.context.span_id for span in runs}


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
def test_sync_run_stream_does_not_set_function_output(mock_template_get):
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
        output_calls = [c for c in span.set_attribute.call_args_list if c[0][0] == "function_output"]
        assert output_calls == []

        chunks = list(stream)

    assert len(chunks) == 2
    output_calls = [c for c in span.set_attribute.call_args_list if c[0][0] == "function_output"]
    assert output_calls == []
    span.end.assert_called_once()


def test_sync_stream_restores_parent_context_between_runs():
    def exercise_hierarchy():
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer(__name__)
        client = PromptLayer(api_key="test_key")
        client.tracer = tracer
        child_names = iter(("LLM call 1", "LLM call 2"))

        def mock_run_internal(**_kwargs):
            return _sync_child_stream(tracer, next(child_names))

        current_span_ids = []
        with patch.object(client, "_run_internal", side_effect=mock_run_internal):
            with tracer.start_as_current_span("wrangler-turn") as root_span:
                root_span_id = root_span.get_span_context().span_id
                list(client.run(prompt_name="test_prompt", stream=True))
                current_span_ids.append(otel_trace.get_current_span().get_span_context().span_id)
                with tracer.start_as_current_span("Tool: demo"):
                    pass
                list(client.run(prompt_name="test_prompt", stream=True))
                current_span_ids.append(otel_trace.get_current_span().get_span_context().span_id)

        spans = exporter.get_finished_spans()
        provider.shutdown()
        return spans, root_span_id, current_span_ids

    _assert_sequential_run_hierarchy(*Context().run(exercise_hierarchy))


@pytest.mark.asyncio
@patch("promptlayer.templates.AsyncTemplateManager.get")
async def test_async_run_stream_does_not_set_function_output(mock_template_get):
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
    assert output_calls == []
    span.end.assert_called_once()


@pytest.mark.asyncio
async def test_async_stream_restores_parent_context_between_runs():
    async def exercise_hierarchy():
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer(__name__)
        client = AsyncPromptLayer(api_key="test_key")
        client.tracer = tracer
        child_names = iter(("LLM call 1", "LLM call 2"))

        async def mock_run_internal(**_kwargs):
            return _async_child_stream(tracer, next(child_names))

        current_span_ids = []
        with patch.object(client, "_run_internal", side_effect=mock_run_internal):
            with tracer.start_as_current_span("wrangler-turn") as root_span:
                root_span_id = root_span.get_span_context().span_id
                async for _chunk in await client.run(prompt_name="test_prompt", stream=True):
                    pass
                current_span_ids.append(otel_trace.get_current_span().get_span_context().span_id)
                with tracer.start_as_current_span("Tool: demo"):
                    pass
                async for _chunk in await client.run(prompt_name="test_prompt", stream=True):
                    pass
                current_span_ids.append(otel_trace.get_current_span().get_span_context().span_id)

        spans = exporter.get_finished_spans()
        provider.shutdown()
        return spans, root_span_id, current_span_ids

    _assert_sequential_run_hierarchy(*await asyncio.create_task(exercise_hierarchy()))
