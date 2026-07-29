import httpx
import pytest
from openai import OpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from promptlayer import PromptLayer
from promptlayer.span_exporter import (
    OpenAIPromptTemplateSpanProcessor,
)


def test_direct_openai_client_emits_root_span_with_prompt_template(monkeypatch):
    instrumentation = pytest.importorskip("opentelemetry.instrumentation.genai.openai")
    instrumentor = instrumentation.OpenAIInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        pytest.skip("OpenAI SDK is already instrumented by this test process")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(OpenAIPromptTemplateSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor.instrument(tracer_provider=provider)

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
                        "message": {
                            "role": "assistant",
                            "content": "Tracing connects work across services.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 6,
                    "total_tokens": 11,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    openai_client = OpenAI(
        api_key="sk-test",
        base_url="https://openai.test/v1",
        http_client=http_client,
    )

    try:
        prompt_blueprint = {
            "id": 42,
            "version": 3,
            "metadata": {"model": {"provider": "openai"}},
            "prompt_template": {"type": "chat"},
            "llm_kwargs": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What is tracing?"}],
            },
        }
        monkeypatch.setattr(
            "promptlayer.templates.get_prompt_template",
            lambda *_args, **_kwargs: prompt_blueprint,
        )
        promptlayer_client = PromptLayer(api_key="pl-test")
        prompt = promptlayer_client.templates.get(
            "support-answer",
            {"label": "production"},
        )

        completion = openai_client.chat.completions.create(**prompt["llm_kwargs"])

        spans = exporter.get_finished_spans()
        assert completion.choices[0].message.content == "Tracing connects work across services."
        assert len(spans) == 1
        assert spans[0].parent is None
        assert spans[0].attributes["gen_ai.provider.name"] == "openai"
        assert spans[0].attributes["promptlayer.prompt.name"] == "support-answer"
        assert spans[0].attributes["promptlayer.prompt.id"] == "42"
        assert spans[0].attributes["promptlayer.prompt.version"] == "3"
        assert spans[0].attributes["promptlayer.prompt.label"] == "production"
    finally:
        openai_client.close()
        instrumentor.uninstrument()
        provider.shutdown()
