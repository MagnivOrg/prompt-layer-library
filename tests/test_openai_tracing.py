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
from opentelemetry.trace import SpanKind, StatusCode

from promptlayer import AsyncPromptLayer, PromptLayer, instrument_openai, tracing
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
    monkeypatch.setattr(tracing, "_instrumented_provider_owners", {})
    original_semconv = os.environ.get("OTEL_SEMCONV_STABILITY_OPT_IN")
    yield
    if original_semconv is None:
        os.environ.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
    else:
        os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = original_semconv


def _install_fake_openai_instrumentor(monkeypatch):
    instrumentor = SimpleNamespace(
        is_instrumented_by_opentelemetry=False,
        instrument=Mock(),
    )

    def instrument(**_kwargs):
        instrumentor.is_instrumented_by_opentelemetry = True

    instrumentor.instrument.side_effect = instrument
    module = SimpleNamespace(OpenAIInstrumentor=lambda: instrumentor)
    monkeypatch.setattr(tracing, "_module_available", lambda module_name: module_name == "openai")
    monkeypatch.setattr(tracing.importlib, "import_module", lambda module_name: module)
    return instrumentor


def _capture_otlp_exporters(monkeypatch):
    exporter_calls = []
    monkeypatch.setattr(
        tracing,
        "OTLPSpanExporter",
        lambda **kwargs: exporter_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(tracing, "BatchSpanProcessor", lambda exporter: exporter)
    return exporter_calls


def test_configure_tracing_instruments_only_openai_once(monkeypatch):
    provider = _FakeTracerProvider()
    instrumentor = _install_fake_openai_instrumentor(monkeypatch)
    exporter_calls = _capture_otlp_exporters(monkeypatch)

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


def test_instrument_openai_is_idempotent_for_application_provider(monkeypatch):
    provider = _FakeTracerProvider()
    instrumentor = _install_fake_openai_instrumentor(monkeypatch)
    exporter_calls = _capture_otlp_exporters(monkeypatch)

    first = instrument_openai(api_key="pl_test", tracer_provider=provider)
    second = instrument_openai(api_key="pl_test", tracer_provider=provider)

    assert first is provider
    assert second is provider
    assert len(provider.processors) == 2
    assert isinstance(provider.processors[0], OpenAIPromptTemplateSpanProcessor)
    instrumentor.instrument.assert_called_once_with(tracer_provider=provider)
    assert len(exporter_calls) == 1


def test_instrument_openai_uses_environment_and_global_provider(monkeypatch):
    provider = _FakeTracerProvider()
    instrumentor = _install_fake_openai_instrumentor(monkeypatch)
    exporter_calls = _capture_otlp_exporters(monkeypatch)
    get_provider = Mock(return_value=provider)
    monkeypatch.setattr(tracing, "_get_or_create_tracer_provider", get_provider)
    monkeypatch.setenv("PROMPTLAYER_API_KEY", "pl_from_env")

    configured_provider = instrument_openai()

    assert configured_provider is provider
    get_provider.assert_called_once_with()
    instrumentor.instrument.assert_called_once_with(tracer_provider=provider)
    assert exporter_calls[0]["headers"]["X-Api-Key"] == "pl_from_env"


def test_instrument_openai_rejects_known_provider_mismatch_before_mutation(monkeypatch):
    first_provider = _FakeTracerProvider()
    second_provider = _FakeTracerProvider()
    _install_fake_openai_instrumentor(monkeypatch)
    _capture_otlp_exporters(monkeypatch)

    instrument_openai(api_key="pl_test", tracer_provider=first_provider)

    with pytest.raises(RuntimeError, match="already instrumented with a different tracer provider"):
        instrument_openai(api_key="pl_test", tracer_provider=second_provider)

    assert second_provider.processors == []


def test_instrument_openai_tracks_declared_provider_for_existing_instrumentor(monkeypatch):
    first_provider = _FakeTracerProvider()
    second_provider = _FakeTracerProvider()
    instrumentor = _install_fake_openai_instrumentor(monkeypatch)
    instrumentor.is_instrumented_by_opentelemetry = True
    _capture_otlp_exporters(monkeypatch)

    instrument_openai(api_key="pl_test", tracer_provider=first_provider)

    with pytest.raises(RuntimeError, match="already instrumented with a different tracer provider"):
        instrument_openai(api_key="pl_test", tracer_provider=second_provider)

    instrumentor.instrument.assert_not_called()
    assert second_provider.processors == []


def test_instrument_openai_requires_optional_dependency_before_mutation(monkeypatch):
    provider = _FakeTracerProvider()
    monkeypatch.setattr(tracing, "_module_available", lambda _module_name: False)

    with pytest.raises(ImportError, match=r"promptlayer\[otel-genai-instrumentation\]"):
        instrument_openai(api_key="pl_test", tracer_provider=provider)

    assert provider.processors == []


def test_instrument_openai_requires_api_key_before_creating_global_provider(monkeypatch):
    get_provider = Mock()
    monkeypatch.delenv("PROMPTLAYER_API_KEY", raising=False)
    monkeypatch.setattr(tracing, "_get_or_create_tracer_provider", get_provider)

    with pytest.raises(ValueError, match="PromptLayer API key not provided"):
        instrument_openai()

    get_provider.assert_not_called()


@pytest.mark.parametrize("provider_name", ["unsupported", "anthropic.bedrock"])
def test_configure_tracing_rejects_unknown_provider(provider_name):
    with pytest.raises(ValueError, match="Unknown tracing provider"):
        tracing.configure_tracing(
            api_key="pl_test",
            tracer_provider=_FakeTracerProvider(),
            providers=(provider_name,),
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
    chat_wrappers = pytest.importorskip("opentelemetry.instrumentation.genai.openai.chat_wrappers")
    response_wrappers = pytest.importorskip("opentelemetry.instrumentation.genai.openai.response_wrappers")
    wrapper_classes = (
        chat_wrappers.ChatStreamWrapper,
        chat_wrappers.AsyncChatStreamWrapper,
        response_wrappers.ResponseStreamWrapper,
        response_wrappers.AsyncResponseStreamWrapper,
        response_wrappers.ResponseStreamManagerWrapper,
        response_wrappers.AsyncResponseStreamManagerWrapper,
    )

    for wrapper_class in wrapper_classes:
        response = type(wrapper_class.__name__, (), {})()
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
    instrumentation = pytest.importorskip("opentelemetry.instrumentation.genai.openai")
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
    monkeypatch.setattr(tracing, "_module_available", lambda module_name: module_name == "openai")

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
    instrumentation = pytest.importorskip("opentelemetry.instrumentation.genai.openai")
    instrumentor = instrumentation.OpenAIInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        pytest.skip("OpenAI SDK is already instrumented by this test process")

    exporter = InMemorySpanExporter()
    monkeypatch.setattr("promptlayer.otlp.OTLPSpanExporter", lambda **kwargs: exporter)
    monkeypatch.setattr(
        "promptlayer.otlp.BatchSpanProcessor",
        lambda configured_exporter: SimpleSpanProcessor(configured_exporter),
    )
    monkeypatch.setattr(tracing, "_module_available", lambda module_name: module_name == "openai")

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
        "prompt_template": {
            "type": "chat",
            "messages": [{"role": "user", "content": "private template"}],
        },
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


def _configure_run_test_client(
    client,
    provider,
    exporter,
    monkeypatch,
    request_function,
    provider_name="openai",
    mock_template=True,
):
    provider.add_span_processor(OpenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client.tracer_provider = provider
    client.tracer = provider.get_tracer("promptlayer.promptlayer")
    blueprint = _promptlayer_run_blueprint()

    def get_prompt(prompt_name, _params=None):
        set_prompt_span_attributes(blueprint, prompt_name, label="production")
        return blueprint

    if mock_template:
        monkeypatch.setattr(client.templates, "get", get_prompt)
    monkeypatch.setattr(
        client,
        "_prepare_llm_data",
        Mock(
            return_value={
                "provider": provider_name,
                "function_name": f"{provider_name}.chat",
                "stream_function": None,
                "request_function": request_function,
                "client_kwargs": {},
                "function_kwargs": blueprint["llm_kwargs"],
                "prompt_blueprint": blueprint,
            }
        ),
    )


@pytest.mark.parametrize(
    ("promptlayer_provider", "span_provider"),
    [
        ("anthropic", "anthropic"),
        ("amazon.bedrock", "aws.bedrock"),
        ("google", "gcp.gemini"),
        ("vertexai", "gcp.vertex_ai"),
    ],
)
def test_promptlayer_run_links_managed_genai_span_to_existing_request_log(
    monkeypatch,
    promptlayer_provider,
    span_provider,
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    client = PromptLayer(api_key="pl-test")
    genai_tracer = provider.get_tracer("opentelemetry.util.genai.handler")
    response = Mock()
    response.model_dump.return_value = {"choices": []}

    def request_function(**_kwargs):
        with genai_tracer.start_as_current_span(
            "chat test-model",
            attributes={
                "gen_ai.provider.name": span_provider,
                "gen_ai.request.model": "test-model",
            },
        ):
            return response

    _configure_run_test_client(
        client,
        provider,
        exporter,
        monkeypatch,
        request_function,
        provider_name=promptlayer_provider,
    )
    track_request = Mock(return_value={"request_id": 17, "prompt_blueprint": {"id": 42}})
    monkeypatch.setattr("promptlayer.promptlayer.track_request", track_request)

    try:
        client.run(
            "support-answer", prompt_version=2, prompt_release_label="production", metadata={"user_id": "user-7"}
        )

        finished_spans = exporter.get_finished_spans()
        assert len(finished_spans) == 3
        spans = {span.name: span for span in finished_spans}
        run_span = spans["PromptLayer Run"]
        fetch_span = spans["Prompt template fetch"]
        genai_span = spans["chat test-model"]
        assert fetch_span.parent.span_id == genai_span.parent.span_id == run_span.context.span_id
        assert run_span.context.trace_id == fetch_span.context.trace_id == genai_span.context.trace_id
        assert run_span.start_time <= fetch_span.start_time <= fetch_span.end_time <= run_span.end_time
        assert run_span.start_time <= genai_span.start_time <= genai_span.end_time <= run_span.end_time
        assert run_span.attributes["node_type"] == "CODE_EXECUTION"
        assert fetch_span.attributes["node_type"] == "PROMPT_TEMPLATE"
        assert genai_span.attributes["node_type"] == "LLM_CALL"
        assert fetch_span.kind == SpanKind.INTERNAL
        assert run_span.attributes["promptlayer.prompt.version"] == "3"
        assert run_span.attributes["promptlayer.metadata.user_id"] == "user-7"
        assert fetch_span.attributes["promptlayer.prompt.requested.version"] == "2"
        for attributes in (run_span.attributes, fetch_span.attributes):
            assert not any(key.startswith(("gen_ai.", "promptlayer.request_log.")) for key in attributes)
            assert {"function_input", "function_output"}.isdisjoint(attributes)
            assert not any(
                secret in repr(dict(attributes)) for secret in ("private template", "Say hello.", "gpt-4o-mini")
            )
        run_span_id = f"{run_span.context.span_id:016x}"
        assert genai_span.attributes["promptlayer.request_log.managed"] is True
        assert genai_span.attributes["promptlayer.request_log.span_id"] == run_span_id
        assert track_request.call_args.kwargs["span_id"] == run_span_id
    finally:
        provider.shutdown()


def test_promptlayer_run_links_managed_openai_span_to_existing_request_log(monkeypatch):
    instrumentation = pytest.importorskip("opentelemetry.instrumentation.genai.openai")
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
    monkeypatch.setattr(tracing, "_module_available", lambda module_name: module_name == "openai")
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
    client = PromptLayer(api_key="pl-test", cache_ttl_seconds=60)
    response = Mock()
    response.model_dump.return_value = {"choices": []}

    _configure_run_test_client(
        client,
        provider,
        exporter,
        monkeypatch,
        lambda **_kwargs: response,
        mock_template=False,
    )
    get_prompt = Mock(return_value=_promptlayer_run_blueprint())
    monkeypatch.setattr("promptlayer.templates.get_prompt_template", get_prompt)
    track_request = Mock(return_value={"request_id": 17, "prompt_blueprint": {"id": 42}})
    monkeypatch.setattr("promptlayer.promptlayer.track_request", track_request)

    try:
        client.run("support-answer")
        client.run("support-answer")
        error = RuntimeError("template unavailable")
        monkeypatch.setattr(client.templates, "get", Mock(side_effect=error))
        with pytest.raises(RuntimeError) as raised:
            client.run("missing-answer")

        finished_spans = exporter.get_finished_spans()
        run_spans = [span for span in finished_spans if span.name == "PromptLayer Run"]
        fetch_spans = [span for span in finished_spans if span.name == "Prompt template fetch"]
        assert raised.value is error
        assert len(run_spans) == len(fetch_spans) == 3
        assert [span.attributes.get("promptlayer.prompt.cache_hit") for span in fetch_spans] == [False, True, None]
        assert get_prompt.call_count == 1
        assert fetch_spans[-1].status.status_code == run_spans[-1].status.status_code == StatusCode.ERROR
        assert fetch_spans[-1].attributes["error.type"] == "RuntimeError"
        assert run_spans[-1].attributes["error.type"] == "RuntimeError"
        assert [call.kwargs["span_id"] for call in track_request.call_args_list] == [
            f"{span.context.span_id:016x}" for span in run_spans[:2]
        ]
    finally:
        provider.shutdown()


def test_promptlayer_run_links_managed_openai_error_span_to_existing_request_log(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    client = PromptLayer(api_key="pl-test")
    openai_tracer = provider.get_tracer("opentelemetry.util.genai.handler")

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
    openai_tracer = provider.get_tracer("opentelemetry.util.genai.handler")
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
        fetch_span = spans["Prompt template fetch"]
        openai_span = spans["chat gpt-4o-mini"]
        assert result["request_id"] == 17
        assert fetch_span.parent.span_id == run_span.context.span_id
        assert openai_span.parent is not None
        assert openai_span.parent.span_id == run_span.context.span_id
        assert openai_span.attributes["promptlayer.request_log.managed"] is True
        run_span_id = f"{run_span.context.span_id:016x}"
        assert track_request.call_args.kwargs["span_id"] == run_span_id
        assert openai_span.attributes["promptlayer.request_log.span_id"] == run_span_id
    finally:
        provider.shutdown()
