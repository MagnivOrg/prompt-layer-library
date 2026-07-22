"""Tests for active Eval tracer context nesting."""

from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from promptlayer.evaluations.tracing import run_case_in_span
from promptlayer.promptlayer import PromptLayer
from promptlayer.tracing_context import active_eval_tracer, resolve_tracer


@pytest.fixture(scope="module")
def in_memory_spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    exporter.clear()


def test_resolve_tracer_prefers_active_eval_tracer():
    fallback = MagicMock(name="fallback")
    active = MagicMock(name="active")
    assert resolve_tracer(fallback) is fallback

    token = active_eval_tracer.set(active)
    try:
        assert resolve_tracer(fallback) is active
        assert resolve_tracer(None) is active
    finally:
        active_eval_tracer.reset(token)

    assert resolve_tracer(fallback) is fallback


def test_promptlayer_tracer_property_resolves_active_eval_tracer(promptlayer_api_key, base_url):
    client = PromptLayer(api_key=promptlayer_api_key, base_url=base_url, enable_tracing=False)
    assert client.tracer is None
    assert client._tracer is None

    active = MagicMock(name="active")
    token = active_eval_tracer.set(active)
    try:
        assert client.tracer is active
    finally:
        active_eval_tracer.reset(token)

    assert client.tracer is None


def test_run_case_in_span_publishes_active_eval_tracer_and_nests_child_spans(in_memory_spans):
    provider, exporter = in_memory_spans
    exporter.clear()
    seen = {}

    def runner(_input):
        seen["active"] = active_eval_tracer.get()
        client_tracer = resolve_tracer(None)
        assert client_tracer is not None
        with client_tracer.start_as_current_span("agent-turn") as child:
            seen["parent"] = child.parent
        return "ok"

    output, trace_id, span_id = run_case_in_span(
        "demo",
        runner,
        {"q": "hi"},
        provider,
        table_id="table-123",
        sheet_id="sheet-456",
    )
    assert output == "ok"
    assert trace_id
    assert span_id
    assert seen["active"] is not None
    assert active_eval_tracer.get() is None

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}
    assert "Eval: demo" in by_name
    assert "agent-turn" in by_name
    eval_span = by_name["Eval: demo"]
    assert eval_span.attributes.get("node_type") == "EVAL"
    assert eval_span.attributes.get("table_id") == "table-123"
    assert eval_span.attributes.get("sheet_id") == "sheet-456"
    child = by_name["agent-turn"]
    assert child.parent is not None
    assert child.parent.span_id == eval_span.context.span_id
    assert child.context.trace_id == eval_span.context.trace_id


def test_promptlayer_traceable_uses_active_eval_tracer_at_call_time(
    in_memory_spans,
    promptlayer_api_key,
    base_url,
):
    provider, exporter = in_memory_spans
    exporter.clear()
    client = PromptLayer(api_key=promptlayer_api_key, base_url=base_url, enable_tracing=False)

    @client.traceable(name="customer-op")
    def work():
        return 1

    def runner(_input):
        return work()

    run_case_in_span("demo-traceable", runner, {}, provider)

    spans = exporter.get_finished_spans()
    names = {span.name for span in spans}
    assert "Eval: demo-traceable" in names
    assert "customer-op" in names
    eval_span = next(span for span in spans if span.name == "Eval: demo-traceable")
    child = next(span for span in spans if span.name == "customer-op")
    assert child.parent is not None
    assert child.parent.span_id == eval_span.context.span_id


def test_promptlayer_trace_tool_sets_tool_execution_attributes(
    in_memory_spans,
    promptlayer_api_key,
    base_url,
):
    provider, exporter = in_memory_spans
    exporter.clear()
    client = PromptLayer(api_key=promptlayer_api_key, base_url=base_url, enable_tracing=False)

    @client.traceTool(name="lookup_account")
    def lookup(email):
        return {"plan": "pro", "email": email}

    def runner(_input):
        return lookup("user@example.com")

    output, _, _ = run_case_in_span("demo-trace-tool", runner, {}, provider)
    assert output == {"plan": "pro", "email": "user@example.com"}

    spans = exporter.get_finished_spans()
    tool_span = next(span for span in spans if span.name == "Tool: lookup_account")
    assert tool_span.attributes.get("node_type") == "CODE_EXECUTION"
    assert tool_span.attributes.get("tool.name") == "lookup_account"
    assert tool_span.attributes.get("function_name") == "lookup_account"


def test_promptlayer_trace_tool_defaults_name_to_function_name(
    in_memory_spans,
    promptlayer_api_key,
    base_url,
):
    provider, exporter = in_memory_spans
    exporter.clear()
    client = PromptLayer(api_key=promptlayer_api_key, base_url=base_url, enable_tracing=False)

    @client.traceTool()
    def send_reset_email(email):
        return {"sent": True, "email": email}

    run_case_in_span("demo-trace-tool-default", lambda _input: send_reset_email("a@b.com"), {}, provider)

    spans = exporter.get_finished_spans()
    tool_span = next(span for span in spans if span.name == "Tool: send_reset_email")
    assert tool_span.attributes.get("node_type") == "CODE_EXECUTION"
    assert tool_span.attributes.get("tool.name") == "send_reset_email"
    assert tool_span.attributes.get("function_name") == "send_reset_email"
