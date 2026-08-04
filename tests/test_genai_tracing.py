import os
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import WeakKeyDictionary

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from promptlayer import tracing
from promptlayer.promptlayer_base import PromptLayerBase
from promptlayer.span_exporter import (
    GenAIPromptTemplateSpanProcessor,
    _mark_genai_request_span,
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


def _install_fake_instrumentors(monkeypatch, provider_names):
    instrumentors = {}
    modules = {}
    sdk_modules = set()

    for provider_name in provider_names:
        config = tracing._PROVIDER_INSTRUMENTATIONS[provider_name]
        sdk_modules.add(config.sdk_module)
        instrumentor = SimpleNamespace(
            is_instrumented_by_opentelemetry=False,
            instrument=Mock(),
        )

        def instrument(*, _instrumentor=instrumentor, **_kwargs):
            _instrumentor.is_instrumented_by_opentelemetry = True

        instrumentor.instrument.side_effect = instrument
        instrumentors[provider_name] = instrumentor
        modules[config.instrumentation_module] = SimpleNamespace(
            **{config.instrumentor_class: lambda value=instrumentor: value}
        )

    monkeypatch.setattr(tracing, "_module_available", lambda module_name: module_name in sdk_modules)
    monkeypatch.setattr(tracing.importlib, "import_module", lambda module_name: modules[module_name])
    return instrumentors


def _capture_otlp_exporters(monkeypatch):
    exporter_calls = []
    monkeypatch.setattr(
        tracing,
        "OTLPSpanExporter",
        lambda **kwargs: exporter_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(tracing, "BatchSpanProcessor", lambda exporter: exporter)
    return exporter_calls


def test_configure_tracing_instruments_anthropic_and_google_once(monkeypatch):
    provider = _FakeTracerProvider()
    instrumentors = _install_fake_instrumentors(monkeypatch, ("anthropic", "google"))
    exporter_calls = _capture_otlp_exporters(monkeypatch)

    first = tracing.configure_tracing(
        api_key="pl_test",
        tracer_provider=provider,
        providers=("anthropic", "google"),
    )
    second = tracing.configure_tracing(
        api_key="pl_test",
        tracer_provider=provider,
        providers=("anthropic", "google"),
    )

    assert first is provider
    assert second is provider
    assert tracing.NATIVE_OTEL_PROVIDERS == ("openai", "anthropic", "google", "bedrock")
    assert len(provider.processors) == 2
    assert isinstance(provider.processors[0], GenAIPromptTemplateSpanProcessor)
    assert len(exporter_calls) == 1
    instrumentors["anthropic"].instrument.assert_called_once_with(tracer_provider=provider)
    instrumentors["google"].instrument.assert_called_once_with(tracer_provider=provider)


def test_configure_tracing_instruments_every_installed_provider_by_default(monkeypatch):
    provider = _FakeTracerProvider()
    instrumentors = _install_fake_instrumentors(monkeypatch, tracing.NATIVE_OTEL_PROVIDERS)
    _capture_otlp_exporters(monkeypatch)

    configured_provider = tracing.configure_tracing(
        api_key="pl_test",
        tracer_provider=provider,
    )

    assert configured_provider is provider
    for provider_name, instrumentor in instrumentors.items():
        expected_kwargs = {"tracer_provider": provider}
        if provider_name == "bedrock":
            expected_kwargs.update(
                request_hook=tracing.bedrock_request_hook,
                response_hook=tracing.bedrock_response_hook,
            )
        instrumentor.instrument.assert_called_once_with(**expected_kwargs)


def test_configure_sdk_instrumentation_respects_provider_selection(monkeypatch):
    provider = _FakeTracerProvider()
    instrumentors = _install_fake_instrumentors(monkeypatch, ("anthropic", "google"))

    tracing._configure_sdk_instrumentation(
        provider,
        providers=("anthropic",),
    )

    instrumentors["anthropic"].instrument.assert_called_once_with(tracer_provider=provider)
    instrumentors["google"].instrument.assert_not_called()


@pytest.mark.parametrize("provider_name", ["bedrock", "amazon.bedrock", "aws.bedrock"])
def test_configure_tracing_accepts_bedrock_provider_aliases(monkeypatch, provider_name):
    provider = _FakeTracerProvider()
    instrumentor = _install_fake_instrumentors(monkeypatch, ("bedrock",))["bedrock"]
    _capture_otlp_exporters(monkeypatch)

    configured_provider = tracing.configure_tracing(
        api_key="pl_test",
        tracer_provider=provider,
        providers=(provider_name,),
    )

    assert configured_provider is provider
    instrumentor.instrument.assert_called_once_with(
        tracer_provider=provider,
        request_hook=tracing.bedrock_request_hook,
        response_hook=tracing.bedrock_response_hook,
    )


def test_implicit_instrumentation_failure_does_not_break_tracing_setup(monkeypatch):
    provider = _FakeTracerProvider()
    instrumentor = _install_fake_instrumentors(monkeypatch, ("anthropic",))["anthropic"]
    instrumentor.instrument.side_effect = RuntimeError("instrumentation failed")

    tracing._configure_sdk_instrumentation(provider, providers=("anthropic",))

    instrumentor.instrument.assert_called_once_with(tracer_provider=provider)
    assert "anthropic" not in tracing._instrumented_provider_owners


def test_explicit_instrumentation_failure_is_still_reported(monkeypatch):
    provider = _FakeTracerProvider()
    instrumentor = _install_fake_instrumentors(monkeypatch, ("anthropic",))["anthropic"]
    instrumentor.instrument.side_effect = RuntimeError("instrumentation failed")

    with pytest.raises(RuntimeError, match="instrumentation failed"):
        tracing._instrument_provider(
            "anthropic",
            provider,
            explicit=True,
            instrumentor=instrumentor,
        )


def test_configure_sdk_instrumentation_accepts_empty_provider_selection():
    provider = _FakeTracerProvider()

    tracing._configure_sdk_instrumentation(provider, providers=())

    assert provider.processors == []


def test_responses_enrichment_failure_does_not_escape_provider_hook(monkeypatch):
    class FailingAttributes(dict):
        def __setitem__(self, key, value):
            raise RuntimeError("attribute failed")

    apply_request_attributes = Mock()
    patch_responses = SimpleNamespace(
        apply_request_attributes=apply_request_attributes,
    )
    monkeypatch.setattr(
        tracing.importlib,
        "import_module",
        lambda _module_name: patch_responses,
    )

    tracing._configure_openai_response_api_enrichment()
    patch_responses.apply_request_attributes(
        SimpleNamespace(attributes=FailingAttributes()),
    )

    apply_request_attributes.assert_called_once()


@pytest.mark.parametrize(
    ("provider_name", "display_name"),
    [
        ("anthropic", "Anthropic"),
        ("google", "Google GenAI"),
        ("bedrock", "AWS Bedrock"),
    ],
)
def test_explicit_provider_dependency_failure_does_not_mutate_provider(
    monkeypatch,
    provider_name,
    display_name,
):
    provider = _FakeTracerProvider()
    monkeypatch.setattr(tracing, "_module_available", lambda _module_name: False)

    with pytest.raises(ImportError, match=display_name):
        tracing.configure_tracing(
            api_key="pl_test",
            tracer_provider=provider,
            providers=(provider_name,),
        )

    assert provider.processors == []


@pytest.mark.parametrize(
    "provider_name",
    [
        "openai",
        "anthropic",
        "aws.bedrock",
        "gemini",
        "vertex_ai",
        "gcp.gemini",
        "gcp.vertex_ai",
    ],
)
def test_genai_prompt_processor_enriches_supported_provider_spans(provider_name):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(GenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    set_prompt_span_attributes(
        {"id": 42, "version": 3},
        "support-answer",
        label="production",
    )
    tracer = provider.get_tracer("opentelemetry.util.genai.handler")
    with tracer.start_as_current_span(
        "chat test-model",
        attributes={"gen_ai.provider.name": provider_name},
    ):
        pass

    attributes = dict(exporter.get_finished_spans()[0].attributes)
    assert attributes["gen_ai.provider.name"] == provider_name
    assert attributes["promptlayer.prompt.name"] == "support-answer"
    assert attributes["promptlayer.prompt.id"] == "42"
    assert attributes["promptlayer.prompt.version"] == "3"
    assert attributes["promptlayer.prompt.label"] == "production"
    provider.shutdown()


@pytest.mark.parametrize(
    ("provider_name", "operation_name", "server_address", "expected_provider", "expected_api"),
    [
        ("openai", "chat", "api.openai.com", "openai", "chat-completions"),
        (
            "openai",
            "chat",
            "example.openai.azure.com",
            "openai.azure",
            "chat-completions",
        ),
        ("anthropic", "chat", "api.anthropic.com", "anthropic", "messages"),
        (
            "anthropic",
            "chat",
            "us-east5-aiplatform.googleapis.com",
            "vertexai",
            "messages",
        ),
        (
            "anthropic",
            "chat",
            "notaiplatform.googleapis.com",
            "anthropic",
            "messages",
        ),
        ("gemini", "generate_content", "generativelanguage.googleapis.com", "google", "generate-content"),
        ("vertex_ai", "generate_content", "aiplatform.googleapis.com", "vertexai", "generate-content"),
        ("gemini", "interactions.create", "generativelanguage.googleapis.com", "google", "interactions"),
        ("gcp.gemini", "embeddings", "generativelanguage.googleapis.com", "google", "embeddings"),
        ("aws.bedrock", "chat", "bedrock-runtime.us-east-1.amazonaws.com", "amazon.bedrock", None),
    ],
)
def test_genai_prompt_processor_sets_canonical_provider_and_api(
    provider_name,
    operation_name,
    server_address,
    expected_provider,
    expected_api,
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(GenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("opentelemetry.util.genai.handler")

    with tracer.start_as_current_span(
        "provider request",
        attributes={
            "gen_ai.provider.name": provider_name,
            "gen_ai.operation.name": operation_name,
            "server.address": server_address,
        },
    ):
        pass

    attributes = dict(exporter.get_finished_spans()[0].attributes)
    assert attributes["promptlayer.provider.type"] == expected_provider
    if expected_api is None:
        assert "promptlayer.api.type" not in attributes
    else:
        assert attributes["promptlayer.api.type"] == expected_api
    provider.shutdown()


@pytest.mark.parametrize(
    ("rpc_method", "expected_api"),
    [
        ("Converse", "converse"),
        ("ConverseStream", "converse"),
        ("InvokeModel", "invoke-model"),
        ("InvokeModelWithResponseStream", "invoke-model"),
    ],
)
def test_genai_prompt_processor_enriches_botocore_bedrock_span(rpc_method, expected_api):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(GenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("opentelemetry.instrumentation.botocore.bedrock-runtime")

    set_prompt_span_attributes(
        {"id": 42, "version": 3},
        "support-answer",
        label="production",
    )
    with tracer.start_as_current_span(
        "Bedrock Runtime request",
        attributes={
            "gen_ai.system": "aws.bedrock",
            "gen_ai.operation.name": "chat",
            "rpc.method": rpc_method,
        },
    ):
        pass

    attributes = dict(exporter.get_finished_spans()[0].attributes)
    assert attributes["gen_ai.provider.name"] == "aws.bedrock"
    assert attributes["promptlayer.provider.type"] == "amazon.bedrock"
    assert attributes["promptlayer.api.type"] == expected_api
    assert attributes["promptlayer.prompt.name"] == "support-answer"
    assert attributes["node_type"] == "LLM_CALL"
    provider.shutdown()


@pytest.mark.parametrize("provider_name", ["anthropic", "gcp.gemini", "gcp.vertex_ai"])
def test_genai_prompt_processor_links_managed_provider_request(provider_name):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(GenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("opentelemetry.util.genai.handler")

    with (
        _mark_genai_request_span(enabled=True, request_log_span_id="abc123"),
        tracer.start_as_current_span(
            "chat test-model",
            attributes={"gen_ai.provider.name": provider_name},
        ),
    ):
        pass

    attributes = dict(exporter.get_finished_spans()[0].attributes)
    assert attributes["promptlayer.request_log.managed"] is True
    assert attributes["promptlayer.request_log.span_id"] == "abc123"
    provider.shutdown()


def test_promptlayer_proxy_links_native_provider_span_to_managed_request(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(GenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    native_tracer = provider.get_tracer("opentelemetry.util.genai.handler")
    response = SimpleNamespace(id="message-test")

    def create_message():
        with native_tracer.start_as_current_span(
            "chat test-model",
            attributes={"gen_ai.provider.name": "anthropic"},
        ):
            return response

    monkeypatch.setattr(
        "promptlayer.promptlayer_base.promptlayer_api_handler",
        lambda *_args, **_kwargs: response,
    )
    proxy = PromptLayerBase(
        api_key="pl-test",
        base_url="https://api.promptlayer.com",
        obj=create_message,
        function_name="anthropic.messages.create",
        provider_type="anthropic",
        tracer=provider.get_tracer("test.promptlayer"),
    )

    assert proxy() is response

    spans_by_name = {span.name: span for span in exporter.get_finished_spans()}
    request_span = spans_by_name["anthropic.messages.create"]
    native_span = spans_by_name["chat test-model"]
    attributes = dict(native_span.attributes)
    assert attributes["promptlayer.request_log.managed"] is True
    assert attributes["promptlayer.request_log.span_id"] == format(
        request_span.context.span_id,
        "016x",
    )
    provider.shutdown()


@pytest.mark.asyncio
async def test_async_promptlayer_proxy_links_native_provider_span_to_managed_request(
    monkeypatch,
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(GenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    native_tracer = provider.get_tracer("opentelemetry.util.genai.handler")
    response = SimpleNamespace(id="response-test")

    async def create_response():
        with native_tracer.start_as_current_span(
            "chat test-model",
            attributes={"gen_ai.provider.name": "openai"},
        ):
            return response

    async def handle_response(*_args, **_kwargs):
        return response

    monkeypatch.setattr(
        "promptlayer.utils.promptlayer_api_handler_async",
        handle_response,
    )
    proxy = PromptLayerBase(
        api_key="pl-test",
        base_url="https://api.promptlayer.com",
        obj=create_response,
        function_name="openai.responses.create",
        provider_type="openai",
        tracer=provider.get_tracer("test.promptlayer"),
    )

    assert await proxy() is response

    spans_by_name = {span.name: span for span in exporter.get_finished_spans()}
    request_span = spans_by_name["openai.responses.create"]
    native_span = spans_by_name["chat test-model"]
    attributes = dict(native_span.attributes)
    assert attributes["promptlayer.request_log.managed"] is True
    assert attributes["promptlayer.request_log.span_id"] == format(
        request_span.context.span_id,
        "016x",
    )
    provider.shutdown()


def test_genai_prompt_processor_ignores_unsupported_provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(GenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_prompt_span_attributes({"id": 42, "version": 3}, "support-answer")
    tracer = provider.get_tracer("opentelemetry.util.genai.handler")

    with tracer.start_as_current_span(
        "chat test-model",
        attributes={"gen_ai.provider.name": "unsupported"},
    ):
        pass

    attributes = dict(exporter.get_finished_spans()[0].attributes)
    assert "promptlayer.prompt.name" not in attributes
    provider.shutdown()


@pytest.mark.parametrize(
    "wrapper_name",
    [
        "AsyncChatStreamWrapper",
        "ResponseStreamManagerWrapper",
        "AsyncResponseStreamManagerWrapper",
        "MessagesStreamWrapper",
        "AsyncMessagesStreamWrapper",
        "MessagesStreamManagerWrapper",
        "AsyncMessagesStreamManagerWrapper",
    ],
)
def test_promptlayer_recognizes_genai_instrumentor_stream_wrappers(wrapper_name):
    response = type(wrapper_name, (), {})()

    result = promptlayer_api_handler(
        api_key="pl-test",
        base_url="https://api.promptlayer.com",
        function_name="provider.stream",
        provider_type="provider",
        args=(),
        kwargs={},
        tags=[],
        response=response,
        request_start_time=1,
        request_end_time=2,
    )

    assert isinstance(result, GeneratorProxy)
    assert result.generator is response


def test_promptlayer_delegates_sync_stream_manager_context(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(GenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    native_tracer = provider.get_tracer("opentelemetry.util.genai.handler")
    final_response = SimpleNamespace(id="resp-test")
    event = SimpleNamespace(type="response.completed", response=final_response)

    class ResponseStream:
        def __init__(self):
            self._events = iter([event])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._events)

        def get_final_response(self):
            return final_response

    class ResponseStreamManagerWrapper:
        def __init__(self):
            self.exited = False

        def __enter__(self):
            with native_tracer.start_as_current_span(
                "chat streamed-model",
                attributes={"gen_ai.provider.name": "openai"},
            ):
                return ResponseStream()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exited = True
            return False

    request = Mock(return_value=17)
    monkeypatch.setattr("promptlayer.utils.promptlayer_api_request", request)
    manager = ResponseStreamManagerWrapper()
    proxy = promptlayer_api_handler(
        api_key="pl-test",
        base_url="https://api.promptlayer.com",
        function_name="openai.responses.create",
        provider_type="openai",
        args=(),
        kwargs={},
        tags=[],
        response=manager,
        request_start_time=1,
        request_end_time=2,
        llm_request_span_id="stream-parent",
    )

    with proxy as stream:
        assert stream is not None
        assert list(stream) == [event]

    assert manager.exited is True
    request.assert_not_called()
    native_span = exporter.get_finished_spans()[0]
    assert native_span.attributes["promptlayer.request_log.managed"] is True
    assert native_span.attributes["promptlayer.request_log.span_id"] == "stream-parent"
    provider.shutdown()


def test_promptlayer_does_not_drain_partially_consumed_stream_manager(monkeypatch):
    first_event = SimpleNamespace(type="response.output_text.delta")

    class ResponseStream:
        def __init__(self):
            self._events = iter([first_event])
            self.get_final_response = Mock(
                side_effect=AssertionError("partial stream must not be drained"),
            )

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._events)

    class ResponseStreamManagerWrapper:
        def __init__(self):
            self.exited = False

        def __enter__(self):
            return ResponseStream()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exited = True
            return False

    request = Mock()
    monkeypatch.setattr("promptlayer.utils.promptlayer_api_request", request)
    manager = ResponseStreamManagerWrapper()
    proxy = promptlayer_api_handler(
        api_key="pl-test",
        base_url="https://api.promptlayer.com",
        function_name="openai.responses.create",
        provider_type="openai",
        args=(),
        kwargs={},
        tags=[],
        response=manager,
        request_start_time=1,
        request_end_time=2,
    )

    with proxy as stream:
        assert next(stream) is first_event

    assert manager.exited is True
    request.assert_not_called()


@pytest.mark.asyncio
async def test_promptlayer_delegates_async_stream_manager_context(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(GenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    native_tracer = provider.get_tracer("opentelemetry.util.genai.handler")
    event = SimpleNamespace(type="content_block_delta")

    class AsyncStream:
        def __init__(self):
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return event

    class AsyncMessagesStreamManagerWrapper:
        def __init__(self):
            self.exited = False

        async def __aenter__(self):
            with native_tracer.start_as_current_span(
                "chat streamed-model",
                attributes={
                    "gen_ai.provider.name": "anthropic",
                    "gen_ai.operation.name": "chat",
                },
            ):
                return AsyncStream()

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self.exited = True
            return False

    request = Mock(return_value=18)
    monkeypatch.setattr("promptlayer.utils.promptlayer_api_request", request)
    manager = AsyncMessagesStreamManagerWrapper()
    proxy = promptlayer_api_handler(
        api_key="pl-test",
        base_url="https://api.promptlayer.com",
        function_name="anthropic.messages.stream",
        provider_type="anthropic",
        args=(),
        kwargs={},
        tags=[],
        response=manager,
        request_start_time=1,
        request_end_time=2,
        llm_request_span_id="async-stream-parent",
    )

    async with proxy as stream:
        assert stream is not None
        assert [item async for item in stream] == [event]

    assert manager.exited is True
    request.assert_not_called()
    native_span = exporter.get_finished_spans()[0]
    assert native_span.attributes["promptlayer.request_log.managed"] is True
    assert native_span.attributes["promptlayer.request_log.span_id"] == "async-stream-parent"
    provider.shutdown()
