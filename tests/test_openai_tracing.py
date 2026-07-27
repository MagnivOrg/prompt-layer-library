import os
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import WeakKeyDictionary

import httpx
import pytest
from openai import OpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from promptlayer import PromptLayer
from promptlayer import tracing
from promptlayer.span_exporter import (
    OpenAIPromptTemplateSpanProcessor,
    set_prompt_span_attributes,
)
from promptlayer.utils import GeneratorProxy, promptlayer_api_handler


class _FakeTracerProvider:
    def __init__(self):
        self.processors = []
        self.tracer = object()

    def add_span_processor(self, processor):
        self.processors.append(processor)

    def get_tracer(self, *args, **kwargs):
        return self.tracer


@pytest.fixture(autouse=True)
def reset_tracing_state(monkeypatch):
    monkeypatch.setattr(tracing, "_exporter_settings", WeakKeyDictionary())
    monkeypatch.setattr(tracing, "_prompt_processor_providers", WeakKeyDictionary())
    original_semconv = os.environ.get("OTEL_SEMCONV_STABILITY_OPT_IN")
    yield
    if original_semconv is None:
        os.environ.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
    else:
        os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = original_semconv


def test_configure_tracing_instruments_only_openai_once(monkeypatch):
    provider = _FakeTracerProvider()
    exporter_calls = []
    instrumentor = SimpleNamespace(
        is_instrumented_by_opentelemetry=False,
        instrument=Mock(),
    )

    def instrument(**kwargs):
        instrumentor.is_instrumented_by_opentelemetry = True

    instrumentor.instrument.side_effect = instrument
    module = SimpleNamespace(OpenAIInstrumentor=lambda: instrumentor)
    monkeypatch.setattr(tracing, "_module_available", lambda module_name: module_name == "openai")
    monkeypatch.setattr(tracing.importlib, "import_module", lambda module_name: module)
    monkeypatch.setattr(
        tracing,
        "OTLPSpanExporter",
        lambda **kwargs: exporter_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(tracing, "BatchSpanProcessor", lambda exporter: exporter)

    first = tracing.configure_tracing(
        api_key="pl_test",
        tracer_provider=provider,
        providers=("openai",),
    )
    second = tracing.configure_tracing(
        api_key="pl_test",
        tracer_provider=provider,
        providers=("openai.azure",),
    )

    assert first is provider
    assert second is provider
    assert len(provider.processors) == 2
    assert isinstance(provider.processors[0], OpenAIPromptTemplateSpanProcessor)
    instrumentor.instrument.assert_called_once_with(tracer_provider=provider)
    assert exporter_calls == [
        {
            "endpoint": "https://api.promptlayer.com/v1/traces",
            "headers": {
                "X-Api-Key": "pl_test",
                "User-Agent": tracing._PROMPTLAYER_USER_AGENT,
                "X-SDK-Version": tracing.SDK_VERSION,
            },
        }
    ]


def test_configure_tracing_rejects_non_openai_provider():
    with pytest.raises(ValueError, match="Unknown tracing provider: anthropic"):
        tracing.configure_tracing(
            api_key="pl_test",
            tracer_provider=_FakeTracerProvider(),
            providers=("anthropic",),
        )


def test_openai_prompt_processor_enriches_native_genai_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(OpenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    set_prompt_span_attributes(
        {"id": 42, "version": 3},
        "support-answer",
        label="production",
    )
    tracer = provider.get_tracer("opentelemetry.util.genai.handler")
    with tracer.start_as_current_span(
        "chat gpt-4o-mini",
        attributes={"gen_ai.provider.name": "openai"},
    ):
        pass

    attributes = dict(exporter.get_finished_spans()[0].attributes)
    assert attributes["promptlayer.prompt.name"] == "support-answer"
    assert attributes["promptlayer.prompt.id"] == "42"
    assert attributes["promptlayer.prompt.version"] == "3"
    assert attributes["promptlayer.prompt.label"] == "production"
    provider.shutdown()


def test_promptlayer_recognizes_instrumented_openai_streams():
    chat_patch = pytest.importorskip("opentelemetry.instrumentation.openai_v2.patch")
    response_wrappers = pytest.importorskip(
        "opentelemetry.instrumentation.openai_v2.response_wrappers"
    )
    wrapper_classes = (
        chat_patch.ChatStreamWrapper,
        chat_patch.LegacyChatStreamWrapper,
        response_wrappers.ResponseStreamWrapper,
        response_wrappers.AsyncResponseStreamWrapper,
    )

    for wrapper_class in wrapper_classes:
        response = object.__new__(wrapper_class)
        result = promptlayer_api_handler(
            api_key="pl-test",
            base_url="https://api.promptlayer.com",
            function_name="openai.chat.completions.create",
            provider_type="openai",
            args=(),
            kwargs={},
            tags=[],
            response=response,
            request_start_time=1,
            request_end_time=2,
        )

        assert isinstance(result, GeneratorProxy)
        assert result.generator is response


def test_promptlayer_enable_tracing_auto_instruments_direct_openai(monkeypatch):
    instrumentation = pytest.importorskip("opentelemetry.instrumentation.openai_v2")
    instrumentor = instrumentation.OpenAIInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        pytest.skip("OpenAI SDK is already instrumented by this test process")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    monkeypatch.setattr(tracing, "OTLPSpanExporter", lambda **kwargs: exporter)
    monkeypatch.setattr(
        tracing,
        "BatchSpanProcessor",
        lambda configured_exporter: SimpleSpanProcessor(configured_exporter),
    )

    def handle_request(_request):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            },
        )

    promptlayer_client = PromptLayer(
        api_key="pl-test",
        enable_tracing=True,
        tracer_provider=provider,
    )
    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    openai_client = OpenAI(
        api_key="sk-test",
        base_url="https://openai.test/v1",
        http_client=http_client,
    )

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello."}],
        )

        spans = exporter.get_finished_spans()
        assert completion.choices[0].message.content == "Hello."
        assert promptlayer_client.tracer_provider is provider
        assert len(spans) == 1
        assert spans[0].parent is None
        assert spans[0].attributes["gen_ai.provider.name"] == "openai"
    finally:
        openai_client.close()
        instrumentor.uninstrument()
        provider.shutdown()
