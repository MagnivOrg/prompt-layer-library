from unittest.mock import Mock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from promptlayer.otlp import (
    create_promptlayer_tracer_provider,
    initialize_promptlayer_tracer,
    resolve_otlp_traces_endpoint,
)
from promptlayer.promptlayer_mixins import PromptLayerMixin
from promptlayer.utils import _PROMPTLAYER_USER_AGENT, SDK_VERSION


class _FakeExporter:
    def __init__(self, seen):
        self.seen = seen

    def __call__(self, *, endpoint, headers):
        self.seen["endpoint"] = endpoint
        self.seen["headers"] = headers
        return self

    def export(self, spans):
        return None

    def shutdown(self):
        return None


def test_resolve_otlp_traces_endpoint_from_base_url():
    assert (
        resolve_otlp_traces_endpoint(base_url="https://api.promptlayer.com/") == "https://api.promptlayer.com/v1/traces"
    )


def test_resolve_otlp_traces_endpoint_prefers_explicit_endpoint(monkeypatch):
    monkeypatch.setenv("PROMPTLAYER_OTLP_TRACES_ENDPOINT", "https://env.example.com/v1/traces")
    assert (
        resolve_otlp_traces_endpoint(
            endpoint="https://override.example.com/custom",
            base_url="https://api.promptlayer.com",
        )
        == "https://override.example.com/custom"
    )


def test_create_promptlayer_tracer_provider_targets_v1_traces(monkeypatch):
    seen = {}
    monkeypatch.setattr("promptlayer.otlp.OTLPSpanExporter", _FakeExporter(seen))

    provider = create_promptlayer_tracer_provider(api_key="pl_test", base_url="https://api.promptlayer.com/")

    assert isinstance(provider, TracerProvider)
    assert seen["endpoint"] == "https://api.promptlayer.com/v1/traces"
    assert seen["headers"] == {
        "X-Api-Key": "pl_test",
        "User-Agent": _PROMPTLAYER_USER_AGENT,
        "X-SDK-Version": SDK_VERSION,
    }
    processors = getattr(getattr(provider, "_active_span_processor", None), "_span_processors", ())
    assert len(processors) == 1
    assert isinstance(processors[0], BatchSpanProcessor)


def test_initialize_tracer_uses_otlp_exporter(monkeypatch):
    seen = {}
    instrument_genai = Mock()
    monkeypatch.setattr("promptlayer.otlp.OTLPSpanExporter", _FakeExporter(seen))
    monkeypatch.setattr(
        "promptlayer.promptlayer_mixins._configure_sdk_instrumentation",
        instrument_genai,
    )

    provider, tracer = PromptLayerMixin._initialize_tracer(
        "pl_test",
        "https://api.promptlayer.com",
        True,
        enable_tracing=True,
    )

    assert provider is not None
    assert tracer is not None
    assert seen["endpoint"] == "https://api.promptlayer.com/v1/traces"
    instrument_genai.assert_called_once_with(provider, providers=None)
    provider.shutdown()


def test_initialize_tracer_uses_explicit_tracer_provider_additively(monkeypatch):
    provider = TracerProvider()
    configure_tracing = Mock(return_value=provider)
    initialize_tracer = Mock()
    monkeypatch.setattr(
        "promptlayer.promptlayer_mixins.configure_tracing",
        configure_tracing,
    )
    monkeypatch.setattr(
        "promptlayer.promptlayer_mixins.initialize_promptlayer_tracer",
        initialize_tracer,
    )

    configured_provider, tracer = PromptLayerMixin._initialize_tracer(
        "pl_test",
        "https://api.promptlayer.com",
        True,
        enable_tracing=True,
        tracer_provider=provider,
        tracing_providers=("openai",),
    )

    assert configured_provider is provider
    assert tracer is not None
    configure_tracing.assert_called_once_with(
        api_key="pl_test",
        base_url="https://api.promptlayer.com",
        tracer_provider=provider,
        providers=("openai",),
    )
    initialize_tracer.assert_not_called()
    provider.shutdown()


def test_initialize_promptlayer_tracer_disabled():
    provider, tracer = initialize_promptlayer_tracer(
        api_key="pl_test",
        base_url="https://api.promptlayer.com",
        enable_tracing=False,
    )
    assert provider is None
    assert tracer is None
