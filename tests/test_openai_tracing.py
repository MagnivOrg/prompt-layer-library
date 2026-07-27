import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from weakref import WeakKeyDictionary

import httpx
import pytest
from openai import OpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from promptlayer import AsyncPromptLayer, PromptLayer, tracing
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
    response_wrappers = pytest.importorskip("opentelemetry.instrumentation.openai_v2.response_wrappers")
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
        assert "promptlayer.request_log.managed" not in spans[0].attributes
        assert "promptlayer.request_log.span_id" not in spans[0].attributes
    finally:
        openai_client.close()
        instrumentor.uninstrument()
        provider.shutdown()


def test_promptlayer_default_tracing_remains_backward_compatible(monkeypatch):
    instrumentation = pytest.importorskip("opentelemetry.instrumentation.openai_v2")
    instrumentor = instrumentation.OpenAIInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        pytest.skip("OpenAI SDK is already instrumented by this test process")

    exporter = InMemorySpanExporter()
    monkeypatch.setattr("promptlayer.otlp.OTLPSpanExporter", lambda **kwargs: exporter)
    monkeypatch.setattr(
        "promptlayer.otlp.BatchSpanProcessor",
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
    )
    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    openai_client = OpenAI(
        api_key="sk-test",
        base_url="https://openai.test/v1",
        http_client=http_client,
    )

    try:
        with promptlayer_client.tracer.start_as_current_span("legacy-promptlayer-span") as parent_span:
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say hello."}],
            )

        spans = exporter.get_finished_spans()
        spans_by_name = {span.name: span for span in spans}
        openai_span = spans_by_name["chat gpt-4o-mini"]
        assert completion.choices[0].message.content == "Hello."
        assert promptlayer_client.tracer_provider is not None
        assert promptlayer_client.tracer_provider.resource.attributes["service.name"] == "prompt-layer-library"
        assert set(spans_by_name) == {"legacy-promptlayer-span", "chat gpt-4o-mini"}
        assert openai_span.parent is not None
        assert openai_span.parent.span_id == parent_span.get_span_context().span_id
        assert openai_span.attributes["gen_ai.provider.name"] == "openai"
    finally:
        openai_client.close()
        instrumentor.uninstrument()
        promptlayer_client.tracer_provider.shutdown()


def _promptlayer_run_blueprint():
    return {
        "id": 42,
        "version": 3,
        "prompt_template": {"type": "chat"},
        "metadata": {
            "model": {
                "provider": "openai",
                "name": "gpt-4o-mini",
                "api_type": "chat-completions",
            }
        },
        "llm_kwargs": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Say hello."}],
        },
    }


def _configure_run_test_client(client, provider, exporter, monkeypatch, request_function):
    provider.add_span_processor(OpenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client.tracer_provider = provider
    client.tracer = provider.get_tracer("promptlayer.promptlayer")
    blueprint = _promptlayer_run_blueprint()

    def get_prompt(prompt_name, _params=None):
        set_prompt_span_attributes(blueprint, prompt_name, label="production")
        return blueprint

    monkeypatch.setattr(client.templates, "get", get_prompt)
    monkeypatch.setattr(
        client,
        "_prepare_llm_data",
        Mock(
            return_value={
                "provider": "openai",
                "function_name": "openai.chat.completions.create",
                "stream_function": None,
                "request_function": request_function,
                "client_kwargs": {},
                "function_kwargs": blueprint["llm_kwargs"],
                "prompt_blueprint": blueprint,
            }
        ),
    )


def test_promptlayer_run_links_managed_openai_span_to_existing_request_log(monkeypatch):
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
    client = PromptLayer(
        api_key="pl-test",
        enable_tracing=True,
        tracer_provider=provider,
    )
    blueprint = _promptlayer_run_blueprint()

    def get_prompt(prompt_name, _params=None):
        set_prompt_span_attributes(blueprint, prompt_name, label="production")
        return blueprint

    monkeypatch.setattr(client.templates, "get", get_prompt)
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
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
        )
    )
    openai_client = OpenAI(
        api_key="sk-test",
        base_url="https://openai.test/v1",
        http_client=http_client,
    )
    monkeypatch.setattr("promptlayer.utils._get_cached_client", lambda *_args, **_kwargs: openai_client)
    track_request = Mock(return_value={"request_id": 17, "prompt_blueprint": {"id": 42}})
    monkeypatch.setattr("promptlayer.promptlayer.track_request", track_request)

    try:
        result = client.run("support-answer")

        spans = {span.name: span for span in exporter.get_finished_spans()}
        run_span = spans["PromptLayer Run"]
        openai_span = spans["chat gpt-4o-mini"]
        assert result["request_id"] == 17
        assert openai_span.parent is not None
        assert openai_span.parent.span_id == run_span.context.span_id
        assert openai_span.attributes["promptlayer.request_log.managed"] is True
        assert openai_span.attributes["promptlayer.prompt.name"] == "support-answer"
        assert openai_span.attributes["promptlayer.prompt.id"] == "42"
        assert openai_span.attributes["promptlayer.prompt.version"] == "3"
        assert "gen_ai.provider.name" not in run_span.attributes
        run_span_id = f"{run_span.context.span_id:016x}"
        assert track_request.call_args.kwargs["span_id"] == run_span_id
        assert openai_span.attributes["promptlayer.request_log.span_id"] == run_span_id
    finally:
        openai_client.close()
        instrumentor.uninstrument()
        provider.shutdown()


def test_promptlayer_run_falls_back_to_run_span_without_openai_span(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    client = PromptLayer(api_key="pl-test")
    response = Mock()
    response.model_dump.return_value = {"choices": []}

    _configure_run_test_client(
        client,
        provider,
        exporter,
        monkeypatch,
        lambda **_kwargs: response,
    )
    track_request = Mock(return_value={"request_id": 17, "prompt_blueprint": {"id": 42}})
    monkeypatch.setattr("promptlayer.promptlayer.track_request", track_request)

    try:
        client.run("support-answer")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "PromptLayer Run"
        assert track_request.call_args.kwargs["span_id"] == f"{spans[0].context.span_id:016x}"
    finally:
        provider.shutdown()


def test_promptlayer_run_links_managed_openai_error_span_to_existing_request_log(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    client = PromptLayer(api_key="pl-test")
    openai_tracer = provider.get_tracer("opentelemetry.instrumentation.openai_v2")

    def request_function(**_kwargs):
        with openai_tracer.start_as_current_span(
            "chat gpt-4o-mini",
            attributes={"gen_ai.provider.name": "openai"},
        ):
            raise RuntimeError("provider failed")

    _configure_run_test_client(client, provider, exporter, monkeypatch, request_function)
    track_request = Mock(return_value={"request_id": 17, "prompt_blueprint": {"id": 42}})
    monkeypatch.setattr("promptlayer.promptlayer.track_request", track_request)

    try:
        with pytest.raises(RuntimeError, match="provider failed"):
            client.run("support-answer")

        spans = {span.name: span for span in exporter.get_finished_spans()}
        run_span = spans["PromptLayer Run"]
        openai_span = spans["chat gpt-4o-mini"]
        assert openai_span.attributes["promptlayer.request_log.managed"] is True
        run_span_id = f"{run_span.context.span_id:016x}"
        assert track_request.call_args.kwargs["span_id"] == run_span_id
        assert openai_span.attributes["promptlayer.request_log.span_id"] == run_span_id
        assert track_request.call_args.kwargs["status"] == "ERROR"
    finally:
        provider.shutdown()


@pytest.mark.asyncio
async def test_async_promptlayer_run_links_managed_openai_span_to_existing_request_log(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    client = AsyncPromptLayer(api_key="pl-test")
    openai_tracer = provider.get_tracer("opentelemetry.instrumentation.openai_v2")
    response = Mock()
    response.model_dump.return_value = {"choices": []}

    async def request_function(**_kwargs):
        with openai_tracer.start_as_current_span(
            "chat gpt-4o-mini",
            attributes={
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
            },
        ):
            return response

    provider.add_span_processor(OpenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client.tracer_provider = provider
    client.tracer = provider.get_tracer("promptlayer.promptlayer")
    blueprint = _promptlayer_run_blueprint()

    async def get_prompt(prompt_name, _params=None):
        set_prompt_span_attributes(blueprint, prompt_name, label="production")
        return blueprint

    monkeypatch.setattr(client.templates, "get", get_prompt)
    monkeypatch.setattr(
        client,
        "_prepare_llm_data",
        Mock(
            return_value={
                "provider": "openai",
                "function_name": "openai.chat.completions.create",
                "stream_function": None,
                "request_function": request_function,
                "client_kwargs": {},
                "function_kwargs": blueprint["llm_kwargs"],
                "prompt_blueprint": blueprint,
            }
        ),
    )
    track_request = AsyncMock(return_value={"request_id": 17, "prompt_blueprint": {"id": 42}})
    monkeypatch.setattr("promptlayer.promptlayer.atrack_request", track_request)

    try:
        result = await client.run("support-answer")

        spans = {span.name: span for span in exporter.get_finished_spans()}
        run_span = spans["PromptLayer Run"]
        openai_span = spans["chat gpt-4o-mini"]
        assert result["request_id"] == 17
        assert openai_span.parent is not None
        assert openai_span.parent.span_id == run_span.context.span_id
        assert openai_span.attributes["promptlayer.request_log.managed"] is True
        run_span_id = f"{run_span.context.span_id:016x}"
        assert track_request.call_args.kwargs["span_id"] == run_span_id
        assert openai_span.attributes["promptlayer.request_log.span_id"] == run_span_id
    finally:
        provider.shutdown()
