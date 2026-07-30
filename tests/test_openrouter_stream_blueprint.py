"""Unit tests for OpenRouter stream blueprint building and chunk accumulation."""

from promptlayer.streaming.blueprint_builder import (
    build_prompt_blueprint_from_openrouter_chunk,
)
from promptlayer.streaming.response_handlers import _accumulate_openrouter_chunks
from promptlayer.streaming.stream_processor import _build_stream_blueprint

METADATA = {"model": {"provider": "openrouter", "name": "openai/gpt-4o-mini", "api_type": "chat"}}


class _Delta:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Choice:
    def __init__(self, delta, finish_reason=None, index=0):
        self.delta = delta
        self.finish_reason = finish_reason
        self.index = index


class _Chunk:
    def __init__(self, choices, **kwargs):
        self.choices = choices
        self.object = "chat.completion.chunk"
        for key, value in kwargs.items():
            setattr(self, key, value)


class _ToolFunction:
    def __init__(self, name="", arguments=""):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id="", function=None, type="function"):
        self.id = id
        self.function = function
        self.type = type


def _content_types(blueprint):
    message = blueprint["prompt_template"]["messages"][0]
    return [c.get("type") for c in message.get("content") or []]


def _content_of_type(blueprint, content_type):
    message = blueprint["prompt_template"]["messages"][0]
    return [c for c in message.get("content") or [] if c.get("type") == content_type]


def test_openrouter_chunk_maps_text_and_tool_calls():
    chunk = _Chunk(
        [
            _Choice(
                _Delta(
                    content="hi",
                    tool_calls=[_ToolCall(id="c1", function=_ToolFunction(name="fn", arguments='{"a":'))],
                )
            )
        ]
    )
    bp = build_prompt_blueprint_from_openrouter_chunk(chunk, METADATA)
    assert _content_of_type(bp, "text")[0]["text"] == "hi"
    tool_calls = bp["prompt_template"]["messages"][0]["tool_calls"]
    assert tool_calls[0]["id"] == "c1"
    assert tool_calls[0]["function"]["name"] == "fn"


def test_openrouter_chunk_maps_reasoning_to_thinking():
    chunk = _Chunk([_Choice(_Delta(reasoning="step 1"))])
    bp = build_prompt_blueprint_from_openrouter_chunk(chunk, METADATA)
    thinking = _content_of_type(bp, "thinking")
    assert len(thinking) == 1
    assert thinking[0]["thinking"] == "step 1"


def test_openrouter_chunk_prefers_reasoning_details_over_duplicate_reasoning():
    """OpenRouter often sends the same token in both ``reasoning`` and details."""
    chunk = {
        "choices": [
            {
                "delta": {
                    "reasoning": "First",
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "First", "id": "r1"},
                    ],
                }
            }
        ]
    }
    bp = build_prompt_blueprint_from_openrouter_chunk(chunk, METADATA)
    thinking = _content_of_type(bp, "thinking")
    assert [t["thinking"] for t in thinking] == ["First"]
    assert thinking[0]["id"] == "r1"


def test_openrouter_chunk_maps_reasoning_details():
    chunk = {
        "choices": [
            {
                "delta": {
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "detail text", "id": "r1", "signature": "sig"},
                        {"type": "reasoning.summary", "summary": "summary text", "id": "r2"},
                        {"type": "reasoning.encrypted", "data": "opaque"},
                    ]
                }
            }
        ]
    }
    bp = build_prompt_blueprint_from_openrouter_chunk(chunk, METADATA)
    thinking = _content_of_type(bp, "thinking")
    assert [t["thinking"] for t in thinking] == ["detail text", "summary text"]
    assert thinking[0]["signature"] == "sig"
    assert thinking[0]["id"] == "r1"


def test_openrouter_chunk_maps_refusal_as_text():
    chunk = _Chunk([_Choice(_Delta(refusal="I can't help with that."))])
    bp = build_prompt_blueprint_from_openrouter_chunk(chunk, METADATA)
    texts = _content_of_type(bp, "text")
    assert texts[0]["text"] == "I can't help with that."


def test_openrouter_chunk_maps_audio_to_output_media():
    chunk = _Chunk(
        [
            _Choice(
                _Delta(
                    audio={
                        "data": "QUJD",
                        "id": "aud_1",
                        "transcript": "hello",
                        "expires_at": 123,
                    }
                )
            )
        ]
    )
    bp = build_prompt_blueprint_from_openrouter_chunk(chunk, METADATA)
    media = _content_of_type(bp, "output_media")
    assert len(media) == 1
    assert media[0]["media_type"] == "audio"
    assert media[0]["mime_type"] == "audio/mpeg"
    assert media[0]["url"].startswith("data:audio/mpeg;base64,")
    assert media[0]["provider_metadata"]["id"] == "aud_1"
    assert media[0]["provider_metadata"]["transcript"] == "hello"


def test_stream_processor_routes_openrouter_to_dedicated_builder():
    chunk = _Chunk([_Choice(_Delta(reasoning="think", content="answer"))])
    bp = _build_stream_blueprint(chunk, METADATA)
    assert _content_types(bp) == ["thinking", "text"]
    assert _content_of_type(bp, "thinking")[0]["thinking"] == "think"
    assert _content_of_type(bp, "text")[0]["text"] == "answer"


def test_accumulate_openrouter_chunks_preserves_extra_fields():
    chunks = [
        {
            "id": "1",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "delta": {
                        "reasoning": "why ",
                        "reasoning_details": [{"type": "reasoning.text", "text": "why "}],
                        "content": "Hello",
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "1",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "delta": {
                        "reasoning": "not",
                        "content": "!",
                        "refusal": "nope",
                        "audio": {"data": "QUJD", "transcript": "hi"},
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]
    result = _accumulate_openrouter_chunks(chunks)
    message = result["choices"][0]["message"]
    assert message["content"] == "Hello!"
    assert message["reasoning"] == "why not"
    assert message["refusal"] == "nope"
    assert message["audio"]["data"] == "QUJD"
    assert len(message["reasoning_details"]) == 1
    assert result["usage"]["total_tokens"] == 2
