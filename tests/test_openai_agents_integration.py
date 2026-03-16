import json

import pytest
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
from promptlayer.utils import SDK_VERSION, _PROMPTLAYER_USER_AGENT

from agents.tracing import set_trace_processors
from agents.tracing.create import function_span, generation_span, trace


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
    assert root_attrs["promptlayer.telemetry.source"] == "openai-agents-python"
    assert root_attrs["openai_agents.trace_id_original"] == trace_id
    assert root_attrs["openai_agents.workflow_name"] == "Weather workflow"
    assert root_attrs["openai_agents.group_id"] == "group-1"
    assert root_attrs["openai_agents.metadata.tenant"] == "acme"
    assert json.loads(root_attrs["openai_agents.metadata_json"]) == {"tenant": "acme", "nested": {"x": 1}}

    attrs = dict(child.attributes)
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
    root = next(span for span in spans if span.name == "Traceparent workflow")
    child = next(span for span in spans if span.name == "Generation")

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
    root = next(span for span in spans if span.name == "No traceparent workflow")

    assert f"{root.context.trace_id:032x}" == "c" * 32
    assert root.parent is None


def test_create_openai_agents_tracer_provider_targets_public_v1_traces(monkeypatch):
    seen = {}

    class FakeExporter:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def export(self, spans):
            return None

        def shutdown(self):
            return None

        def force_flush(self, timeout_millis=30000):
            return True

    monkeypatch.setattr(
        "promptlayer.integrations.openai_agents.instrumentation.OTLPSpanExporter",
        FakeExporter,
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

    class FakeExporter:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def export(self, spans):
            return None

        def shutdown(self):
            return None

        def force_flush(self, timeout_millis=30000):
            return True

    monkeypatch.setattr(
        "promptlayer.integrations.openai_agents.instrumentation.OTLPSpanExporter",
        FakeExporter,
    )

    create_openai_agents_tracer_provider(api_key="pl_test", endpoint="https://collector.example.com/custom-traces")

    assert seen["endpoint"] == "https://collector.example.com/custom-traces"
