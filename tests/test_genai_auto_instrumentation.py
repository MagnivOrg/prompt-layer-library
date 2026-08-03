import json
import os
from weakref import WeakKeyDictionary

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from promptlayer import tracing
from promptlayer.span_exporter import set_prompt_span_attributes


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


def _configure_in_memory_export(monkeypatch, provider_name):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    monkeypatch.setattr(tracing, "OTLPSpanExporter", lambda **_kwargs: exporter)
    monkeypatch.setattr(
        tracing,
        "BatchSpanProcessor",
        lambda configured_exporter: SimpleSpanProcessor(configured_exporter),
    )
    tracing.configure_tracing(
        api_key="pl-test",
        tracer_provider=provider,
        providers=(provider_name,),
    )
    return provider, exporter


def _set_test_prompt():
    set_prompt_span_attributes(
        {"id": 42, "version": 3},
        "support-answer",
        label="production",
    )


def _assert_prompt_attributes(span):
    assert span.attributes["promptlayer.prompt.name"] == "support-answer"
    assert span.attributes["promptlayer.prompt.id"] == "42"
    assert span.attributes["promptlayer.prompt.version"] == "3"
    assert span.attributes["promptlayer.prompt.label"] == "production"


class _FailingExporter(SpanExporter):
    def __init__(self):
        self.export_calls = 0

    def export(self, spans):
        self.export_calls += 1
        return SpanExportResult.FAILURE


def test_failed_trace_export_does_not_affect_openai_response(monkeypatch):
    openai = pytest.importorskip("openai")
    instrumentation = pytest.importorskip("opentelemetry.instrumentation.genai.openai")
    instrumentor = instrumentation.OpenAIInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        pytest.skip("OpenAI SDK is already instrumented by this test process")

    exporter = _FailingExporter()
    provider = TracerProvider()
    monkeypatch.setattr(tracing, "OTLPSpanExporter", lambda **_kwargs: exporter)
    tracing.configure_tracing(
        api_key="pl-test",
        tracer_provider=provider,
        providers=("openai",),
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
                        "message": {
                            "role": "assistant",
                            "content": "Hello.",
                        },
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

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    client = openai.OpenAI(
        api_key="sk-test",
        base_url="https://openai.test/v1",
        http_client=http_client,
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello."}],
        )

        assert completion.choices[0].message.content == "Hello."
        assert provider.force_flush() is True
        assert exporter.export_calls == 1
    finally:
        client.close()
        instrumentor.uninstrument()
        provider.shutdown()


def test_openai_responses_api_emits_enriched_span(monkeypatch):
    openai = pytest.importorskip("openai")
    instrumentation = pytest.importorskip("opentelemetry.instrumentation.genai.openai")
    instrumentor = instrumentation.OpenAIInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        pytest.skip("OpenAI SDK is already instrumented by this test process")

    provider, exporter = _configure_in_memory_export(monkeypatch, "openai")

    def handle_request(_request):
        return httpx.Response(
            200,
            json={
                "id": "resp-test",
                "created_at": 1.0,
                "model": "gpt-4o-mini",
                "object": "response",
                "output": [
                    {
                        "id": "msg-test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "annotations": [],
                                "logprobs": [],
                                "text": "Hello.",
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "temperature": 1.0,
                "top_p": 1.0,
                "usage": {
                    "input_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 3,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    client = openai.OpenAI(
        api_key="sk-test",
        base_url="https://openai.test/v1",
        http_client=http_client,
    )

    try:
        _set_test_prompt()
        response = client.responses.create(model="gpt-4o-mini", input="Say hello.")

        spans = exporter.get_finished_spans()
        assert response.output_text == "Hello."
        assert len(spans) == 1
        assert spans[0].attributes["gen_ai.provider.name"] == "openai"
        assert spans[0].attributes["gen_ai.operation.name"] == "chat"
        assert spans[0].attributes["promptlayer.provider.type"] == "openai"
        assert spans[0].attributes["promptlayer.api.type"] == "responses"
        _assert_prompt_attributes(spans[0])
    finally:
        client.close()
        instrumentor.uninstrument()
        provider.shutdown()


def test_anthropic_messages_api_emits_enriched_span(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    instrumentation = pytest.importorskip("opentelemetry.instrumentation.genai.anthropic")
    instrumentor = instrumentation.AnthropicInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        pytest.skip("Anthropic SDK is already instrumented by this test process")

    provider, exporter = _configure_in_memory_export(monkeypatch, "anthropic")

    def handle_request(_request):
        return httpx.Response(
            200,
            json={
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [{"type": "text", "text": "Hello."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    client = anthropic.Anthropic(
        api_key="sk-ant-test",
        base_url="https://anthropic.test",
        http_client=http_client,
    )

    try:
        _set_test_prompt()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=32,
            messages=[{"role": "user", "content": "Say hello."}],
        )

        spans = exporter.get_finished_spans()
        assert response.content[0].text == "Hello."
        assert len(spans) == 1
        assert spans[0].attributes["gen_ai.provider.name"] == "anthropic"
        assert spans[0].attributes["gen_ai.operation.name"] == "chat"
        assert spans[0].attributes["promptlayer.provider.type"] == "anthropic"
        assert spans[0].attributes["promptlayer.api.type"] == "messages"
        _assert_prompt_attributes(spans[0])
    finally:
        client.close()
        instrumentor.uninstrument()
        provider.shutdown()


def test_bedrock_converse_emits_enriched_span(monkeypatch):
    boto3 = pytest.importorskip("boto3")
    botocore_stub = pytest.importorskip("botocore.stub")
    instrumentation = pytest.importorskip("opentelemetry.instrumentation.botocore")
    instrumentor = instrumentation.BotocoreInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        pytest.skip("Botocore SDK is already instrumented by this test process")

    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY")
    provider, exporter = _configure_in_memory_export(monkeypatch, "bedrock")
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    request = {
        "modelId": "global.anthropic.claude-sonnet-5",
        "messages": [
            {
                "role": "user",
                "content": [{"text": "Say hello."}],
            }
        ],
        "inferenceConfig": {"maxTokens": 32},
    }
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello."}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 2,
            "outputTokens": 1,
            "totalTokens": 3,
        },
        "metrics": {"latencyMs": 1},
        "ResponseMetadata": {
            "RequestId": "bedrock-test-request",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {},
            "RetryAttempts": 0,
        },
    }

    try:
        _set_test_prompt()
        with botocore_stub.Stubber(client) as stubber:
            stubber.add_response("converse", response, request)
            result = client.converse(**request)

        spans = exporter.get_finished_spans()
        assert result["output"]["message"]["content"][0]["text"] == "Hello."
        assert len(spans) == 1
        assert spans[0].name == "chat global.anthropic.claude-sonnet-5"
        assert spans[0].attributes["gen_ai.provider.name"] == "aws.bedrock"
        assert spans[0].attributes["gen_ai.operation.name"] == "chat"
        assert spans[0].attributes["gen_ai.usage.input_tokens"] == 2
        assert spans[0].attributes["gen_ai.usage.output_tokens"] == 1
        assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
            {
                "role": "user",
                "parts": [{"type": "text", "content": "Say hello."}],
            }
        ]
        assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "Hello."}],
                "finish_reason": "end_turn",
            }
        ]
        assert spans[0].attributes["promptlayer.provider.type"] == "amazon.bedrock"
        assert spans[0].attributes["promptlayer.api.type"] == "converse"
        assert spans[0].attributes["node_type"] == "LLM_CALL"
        _assert_prompt_attributes(spans[0])
    finally:
        instrumentor.uninstrument()
        provider.shutdown()


@pytest.mark.parametrize(
    ("vertexai", "expected_providers"),
    [
        (False, {"gemini", "gcp.gemini"}),
        (True, {"vertex_ai", "gcp.vertex_ai"}),
    ],
)
def test_google_generate_content_emits_enriched_span(
    monkeypatch,
    vertexai,
    expected_providers,
):
    genai = pytest.importorskip("google.genai")
    google_types = pytest.importorskip("google.genai.types")
    google_models = pytest.importorskip("google.genai.models")
    instrumentation = pytest.importorskip("opentelemetry.instrumentation.google_genai")
    instrumentor = instrumentation.GoogleGenAiSdkInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        pytest.skip("Google GenAI SDK is already instrumented by this test process")

    def generate_content(_self, *, model, contents, config=None):
        del model, contents, config
        return google_types.GenerateContentResponse(
            candidates=[
                google_types.Candidate(
                    content=google_types.Content(
                        role="model",
                        parts=[google_types.Part(text="Hello.")],
                    ),
                    finish_reason=google_types.FinishReason.STOP,
                )
            ],
            usage_metadata=google_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=2,
                candidates_token_count=1,
                total_token_count=3,
            ),
        )

    monkeypatch.setattr(google_models.Models, "generate_content", generate_content)
    provider, exporter = _configure_in_memory_export(monkeypatch, "google")
    if vertexai:
        google_auth = pytest.importorskip("google.auth.credentials")
        client = genai.Client(
            vertexai=True,
            project="test-project",
            location="us-central1",
            credentials=google_auth.AnonymousCredentials(),
        )
    else:
        client = genai.Client(api_key="google-test")

    try:
        _set_test_prompt()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hello.",
        )

        spans = exporter.get_finished_spans()
        assert response.text == "Hello."
        assert len(spans) == 1
        assert spans[0].attributes["gen_ai.provider.name"] in expected_providers
        assert spans[0].attributes["gen_ai.operation.name"] == "generate_content"
        assert spans[0].attributes["promptlayer.provider.type"] == ("vertexai" if vertexai else "google")
        assert spans[0].attributes["promptlayer.api.type"] == "generate-content"
        _assert_prompt_attributes(spans[0])
    finally:
        client.close()
        instrumentor.uninstrument()
        provider.shutdown()
