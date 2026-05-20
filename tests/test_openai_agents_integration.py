import json

import pytest
from agents.tracing import set_trace_processors
from agents.tracing.create import agent_span, function_span, generation_span, response_span, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext
from opentelemetry.trace.status import StatusCode

from promptlayer.integrations.openai_agents import (
    OpenAIAgentsTracingProviderError,
    PromptLayerOpenAIAgentsProcessor,
    create_openai_agents_tracer_provider,
    instrument_openai_agents,
)
from promptlayer.integrations.openai_agents.ids import map_span_id, map_trace_id
from promptlayer.integrations.openai_agents.mapping import normalize_response_items
from promptlayer.utils import _PROMPTLAYER_USER_AGENT, SDK_VERSION


@pytest.fixture(autouse=True)
def reset_agents_trace_processors():
    set_trace_processors([])
    yield
    set_trace_processors([])


@pytest.fixture
def in_memory_tracer_provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    provider.shutdown()


def _finished_spans(exporter: InMemorySpanExporter):
    spans = exporter.get_finished_spans()
    assert spans
    return spans


def _find_root_and_child(spans):
    root = next(span for span in spans if span.parent is None)
    child = next(span for span in spans if span.parent is not None)
    return root, child


class _FakeExporter:
    def __init__(self, seen: dict):
        self._seen = seen

    def __call__(self, **kwargs):
        self._seen.update(kwargs)
        return self

    def export(self, spans):
        return None

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis=30000):
        return True


def test_instrument_openai_agents_rejects_non_sdk_provider():
    with pytest.raises(OpenAIAgentsTracingProviderError, match="TracerProvider"):
        instrument_openai_agents(tracer_provider=object())


def test_generation_span_emits_canonical_attrs_and_deterministic_ids(in_memory_tracer_provider):
    provider, exporter = in_memory_tracer_provider
    processor = PromptLayerOpenAIAgentsProcessor(tracer_provider=provider)
    set_trace_processors([processor])

    trace_id = "trace_" + ("a" * 32)
    span_id = "span_" + ("b" * 24)

    with trace(
        "Weather workflow", trace_id=trace_id, group_id="group-1", metadata={"tenant": "acme", "nested": {"x": 1}}
    ):
        with generation_span(
            input=[{"role": "user", "content": "What is the weather?"}],
            output=[
                {
                    "role": "assistant",
                    "content": "Calling weather",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "weather",
                            "arguments": {"city": "Tokyo"},
                        }
                    ],
                }
            ],
            model="gpt-4.1",
            model_config={"temperature": 0.2},
            usage={"input_tokens": 7, "output_tokens": 3},
            span_id=span_id,
        ):
            pass

    spans = _finished_spans(exporter)
    root, child = _find_root_and_child(spans)

    assert f"{root.context.trace_id:032x}" == map_trace_id(trace_id)
    assert f"{child.context.trace_id:032x}" == map_trace_id(trace_id)
    assert f"{child.context.span_id:016x}" == map_span_id(span_id)

    root_attrs = dict(root.attributes)
    assert root.name == "LLM Session"
    assert root_attrs["node_type"] == "LLM_SESSION"
    assert root_attrs["promptlayer.telemetry.source"] == "openai-agents-python"
    assert root_attrs["openai_agents.trace_id_original"] == trace_id
    assert root_attrs["openai_agents.workflow_name"] == "Weather workflow"
    assert root_attrs["openai_agents.group_id"] == "group-1"
    assert root_attrs["openai_agents.metadata.tenant"] == "acme"
    assert json.loads(root_attrs["openai_agents.metadata_json"]) == {"tenant": "acme", "nested": {"x": 1}}

    attrs = dict(child.attributes)
    assert child.name == "LLM call"
    assert attrs["node_type"] == "LLM_CALL"
    assert attrs["promptlayer.telemetry.source"] == "openai-agents-python"
    assert attrs["openai_agents.span_type"] == "generation"
    assert attrs["gen_ai.provider.name"] == "openai.responses"
    assert attrs["gen_ai.request.model"] == "gpt-4.1"
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 3
    assert attrs["gen_ai.prompt.0.role"] == "user"
    assert attrs["gen_ai.prompt.0.content"] == "What is the weather?"
    assert attrs["gen_ai.completion.0.role"] == "assistant"
    assert attrs["gen_ai.completion.0.content"] == "Calling weather"
    assert json.loads(attrs["gen_ai.completion.0.tool_calls"]) == [
        {"id": "call_1", "type": "tool_call", "name": "weather", "arguments": {"city": "Tokyo"}}
    ]
    assert json.loads(attrs["openai_agents.model_config_json"]) == {"temperature": 0.2}
    assert json.loads(attrs["openai_agents.generation.raw_input_json"]) == [
        {"role": "user", "content": "What is the weather?"}
    ]


def test_agent_span_emits_llm_session_semantics(in_memory_tracer_provider):
    provider, exporter = in_memory_tracer_provider
    processor = PromptLayerOpenAIAgentsProcessor(tracer_provider=provider)
    set_trace_processors([processor])

    with trace("Agent workflow"):
        with agent_span(
            name="PromptLayer Demo Agent",
            handoffs=[],
            tools=[],
            output_type="str",
        ):
            pass

    spans = _finished_spans(exporter)
    _, child = _find_root_and_child(spans)
    attrs = dict(child.attributes)

    assert child.name == "LLM Session"
    assert attrs["node_type"] == "LLM_SESSION"
    assert attrs["openai_agents.span_type"] == "agent"
    assert attrs["openai_agents.agent.name"] == "PromptLayer Demo Agent"
    assert attrs["openai_agents.agent.output_type"] == "str"
    assert json.loads(attrs["openai_agents.agent.handoffs_json"]) == []
    assert json.loads(attrs["openai_agents.agent.tools_json"]) == []


def test_response_span_emits_openai_reasoning_as_genai_thinking(in_memory_tracer_provider):
    provider, exporter = in_memory_tracer_provider
    processor = PromptLayerOpenAIAgentsProcessor(tracer_provider=provider)
    set_trace_processors([processor])

    response = {
        "id": "resp_123",
        "object": "response",
        "model": "gpt-5-2025-08-07",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_123",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "I should compute the multiplication directly.",
                    }
                ],
                "encrypted_content": "enc",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "19 times 23 is 437."}],
            },
        ],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 80,
            "output_tokens_details": {"reasoning_tokens": 64},
        },
    }

    with trace("Reasoning workflow"):
        with response_span(
            response=response,
            span_id="span_" + ("d" * 24),
        ):
            pass

    spans = _finished_spans(exporter)
    child = next(span for span in spans if span.name == "LLM call")
    attrs = dict(child.attributes)

    assert attrs["node_type"] == "LLM_CALL"
    assert attrs["gen_ai.provider.name"] == "openai.responses"
    assert attrs["gen_ai.request.model"] == "gpt-5-2025-08-07"
    assert attrs["gen_ai.response.id"] == "resp_123"
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 80
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 64
    assert attrs["gen_ai.completion.0.role"] == "assistant"
    assert attrs["gen_ai.completion.0.thinking"] == "I should compute the multiplication directly."
    assert attrs["gen_ai.completion.1.role"] == "assistant"
    assert attrs["gen_ai.completion.1.content"] == "19 times 23 is 437."
    assert json.loads(attrs["openai_agents.response.raw_json"]) == response


def test_function_span_stays_namespaced_without_genai_attrs(in_memory_tracer_provider):
    provider, exporter = in_memory_tracer_provider
    processor = PromptLayerOpenAIAgentsProcessor(tracer_provider=provider)
    set_trace_processors([processor])

    with trace("Function workflow"):
        with function_span(
            name="weather_lookup",
            input='{"city":"Tokyo"}',
            output={"forecast": "sunny"},
        ):
            pass

    spans = _finished_spans(exporter)
    _, child = _find_root_and_child(spans)
    attrs = dict(child.attributes)

    assert attrs["openai_agents.span_type"] == "function"
    assert attrs["node_type"] == "CODE_EXECUTION"
    assert attrs["tool_name"] == "weather_lookup"
    assert attrs["function_input"] == '{"city":"Tokyo"}'
    assert attrs["function_output"] == '{"forecast": "sunny"}'
    assert attrs["openai_agents.function.name"] == "weather_lookup"
    assert attrs["openai_agents.function.input"] == '{"city":"Tokyo"}'
    assert json.loads(attrs["openai_agents.function.output_json"]) == {"forecast": "sunny"}
    assert not any(key.startswith("gen_ai.") for key in attrs)


def test_generation_span_records_error_status_and_exception_event(in_memory_tracer_provider):
    provider, exporter = in_memory_tracer_provider
    processor = PromptLayerOpenAIAgentsProcessor(tracer_provider=provider)
    set_trace_processors([processor])

    with trace("Error workflow"):
        span = generation_span(
            input=[{"role": "user", "content": "Hi"}],
            model="gpt-4.1",
        )
        span.start()
        span.set_error({"message": "boom", "data": {"code": "bad_request"}})
        span.finish()

    spans = _finished_spans(exporter)
    _, child = _find_root_and_child(spans)

    assert child.status.status_code is StatusCode.ERROR
    assert child.status.description == "boom"
    exception_event = next(event for event in child.events if event.name == "exception")
    assert exception_event.attributes["exception.type"] == "OpenAIAgentsError"
    assert exception_event.attributes["exception.message"] == "boom"
    assert json.loads(exception_event.attributes["openai_agents.error_json"]) == {
        "message": "boom",
        "data": {"code": "bad_request"},
    }


def test_traceparent_metadata_parents_the_synthetic_root(in_memory_tracer_provider):
    provider, exporter = in_memory_tracer_provider
    processor = PromptLayerOpenAIAgentsProcessor(tracer_provider=provider)
    set_trace_processors([processor])

    traceparent = "00-11111111111111111111111111111111-2222222222222222-01"
    agents_trace_id = "trace_" + ("a" * 32)

    with trace(
        "Traceparent workflow",
        trace_id=agents_trace_id,
        metadata={"traceparent": traceparent, "tenant": "acme"},
    ):
        with generation_span(
            input=[{"role": "user", "content": "hi"}],
            output=[{"role": "assistant", "content": "hello"}],
            model="gpt-4.1",
        ):
            pass

    spans = _finished_spans(exporter)
    root = next(span for span in spans if span.parent is not None and span.name == "LLM Session")
    child = next(span for span in spans if span.name == "LLM call")

    assert f"{root.context.trace_id:032x}" == "11111111111111111111111111111111"
    assert f"{child.context.trace_id:032x}" == "11111111111111111111111111111111"
    assert isinstance(root.parent, SpanContext)
    assert f"{root.parent.trace_id:032x}" == "11111111111111111111111111111111"
    assert f"{root.parent.span_id:016x}" == "2222222222222222"
    assert dict(root.attributes)["openai_agents.trace_id_original"] == agents_trace_id


def test_active_local_context_does_not_override_agents_trace_id_without_traceparent(in_memory_tracer_provider):
    provider, exporter = in_memory_tracer_provider
    processor = PromptLayerOpenAIAgentsProcessor(tracer_provider=provider)
    set_trace_processors([processor])

    upstream_tracer = provider.get_tracer("upstream")

    with upstream_tracer.start_as_current_span("upstream"):
        with trace("No traceparent workflow", trace_id="trace_" + ("c" * 32)):
            with generation_span(
                input=[{"role": "user", "content": "hi"}],
                output=[{"role": "assistant", "content": "hello"}],
                model="gpt-4.1",
            ):
                pass

    spans = _finished_spans(exporter)
    root = next(span for span in spans if span.name == "LLM Session")

    assert f"{root.context.trace_id:032x}" == "c" * 32
    assert root.parent is None


def test_create_openai_agents_tracer_provider_targets_public_v1_traces(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        "promptlayer.integrations.openai_agents.instrumentation.OTLPSpanExporter",
        _FakeExporter(seen),
    )

    provider = create_openai_agents_tracer_provider(api_key="pl_test", base_url="https://api.promptlayer.com/")

    assert isinstance(provider, TracerProvider)
    assert seen["endpoint"] == "https://api.promptlayer.com/v1/traces"
    assert seen["headers"] == {
        "X-Api-Key": "pl_test",
        "User-Agent": _PROMPTLAYER_USER_AGENT,
        "X-SDK-Version": SDK_VERSION,
    }


def test_create_openai_agents_tracer_provider_allows_endpoint_override(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        "promptlayer.integrations.openai_agents.instrumentation.OTLPSpanExporter",
        _FakeExporter(seen),
    )

    create_openai_agents_tracer_provider(api_key="pl_test", endpoint="https://collector.example.com/custom-traces")

    assert seen["endpoint"] == "https://collector.example.com/custom-traces"


def test_normalize_response_items_keeps_message_like_inputs_without_type():
    items = [
        {
            "role": "user",
            "content": "I'm planning a trip to Tokyo. What should I know?",
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "weather_lookup",
            "arguments": '{"city":"Tokyo"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "{'temp_c': 24, 'condition': 'Sunny'}",
        },
    ]

    assert normalize_response_items(items) == [
        {
            "role": "user",
            "content": "I'm planning a trip to Tokyo. What should I know?",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "tool_call",
                    "name": "weather_lookup",
                    "arguments": {"city": "Tokyo"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "{'temp_c': 24, 'condition': 'Sunny'}",
        },
    ]


def test_normalize_response_items_preserves_openai_reasoning_summary():
    items = [
        {
            "type": "reasoning",
            "id": "rs_123",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "I should compute the multiplication directly.",
                },
                {
                    "type": "summary_text",
                    "text": "The answer is 437.",
                },
            ],
            "encrypted_content": "enc",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "19 times 23 is 437."}],
        },
    ]

    assert normalize_response_items(items) == [
        {
            "role": "assistant",
            "thinking": "I should compute the multiplication directly.\n\nThe answer is 437.",
        },
        {
            "role": "assistant",
            "content": "19 times 23 is 437.",
        },
    ]
