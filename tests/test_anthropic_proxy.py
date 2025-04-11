import pytest
from anthropic.types import (
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
)
from anthropic.types.raw_message_delta_event import Delta
from anthropic.types.usage import Usage

from tests.utils.vcr import assert_played


def test_anthropic_chat_completion(capsys, anthropic_client):
    with assert_played("test_anthropic_chat_completion.yaml"):
        completion = anthropic_client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is the capital of the United States?"}],
            model="claude-3-haiku-20240307",
        )

    captured = capsys.readouterr()
    assert "WARNING: While" not in captured.err
    assert completion.content is not None


def test_anthropic_chat_completion_with_pl_id(capsys, anthropic_client):
    with assert_played("test_anthropic_chat_completion_with_pl_id.yaml"):
        completion, pl_id = anthropic_client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            model="claude-3-haiku-20240307",
            return_pl_id=True,
        )

    assert "WARNING: While" not in capsys.readouterr().err
    assert completion.content is not None
    assert isinstance(pl_id, int)


def test_anthropic_chat_completion_with_stream(capsys, anthropic_client):
    with assert_played("test_anthropic_chat_completion_with_stream.yaml"):
        completions_gen = anthropic_client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is the capital of Germany?"}],
            model="claude-3-haiku-20240307",
            stream=True,
        )

        completions = [completion for completion in completions_gen]
        assert completions == [
            RawMessageStartEvent(
                message=Message(
                    id="msg_01PP15qoPAWehXLzXosCnnVP",
                    content=[],
                    model="claude-3-haiku-20240307",
                    role="assistant",
                    stop_reason=None,
                    stop_sequence=None,
                    type="message",
                    usage=Usage(
                        cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=14, output_tokens=4
                    ),
                ),
                type="message_start",
            ),
            RawContentBlockStartEvent(
                content_block=TextBlock(citations=None, text="", type="text"), index=0, type="content_block_start"
            ),
            RawContentBlockDeltaEvent(
                delta=TextDelta(text="The capital of Germany", type="text_delta"), index=0, type="content_block_delta"
            ),
            RawContentBlockDeltaEvent(
                delta=TextDelta(text=" is Berlin.", type="text_delta"), index=0, type="content_block_delta"
            ),
            RawContentBlockStopEvent(index=0, type="content_block_stop"),
            RawMessageDeltaEvent(
                delta=Delta(stop_reason="end_turn", stop_sequence=None),
                type="message_delta",
                usage=MessageDeltaUsage(output_tokens=10),
            ),
            RawMessageStopEvent(type="message_stop"),
        ]

    assert "WARNING: While" not in capsys.readouterr().err


def test_anthropic_chat_completion_with_stream_and_pl_id(anthropic_client):
    with assert_played("test_anthropic_chat_completion_with_stream_and_pl_id.yaml"):
        completions_gen = anthropic_client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is the capital of Italy?"}],
            model="claude-3-haiku-20240307",
            stream=True,
            return_pl_id=True,
        )
        completions = [completion for completion, _ in completions_gen]
        assert completions == [
            RawMessageStartEvent(
                message=Message(
                    id="msg_016q2kSZ82qtDP2CNUAKSfLV",
                    content=[],
                    model="claude-3-haiku-20240307",
                    role="assistant",
                    stop_reason=None,
                    stop_sequence=None,
                    type="message",
                    usage=Usage(
                        cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=14, output_tokens=4
                    ),
                ),
                type="message_start",
            ),
            RawContentBlockStartEvent(
                content_block=TextBlock(citations=None, text="", type="text"), index=0, type="content_block_start"
            ),
            RawContentBlockDeltaEvent(
                delta=TextDelta(text="The capital of Italy", type="text_delta"), index=0, type="content_block_delta"
            ),
            RawContentBlockDeltaEvent(
                delta=TextDelta(text=" is Rome.", type="text_delta"), index=0, type="content_block_delta"
            ),
            RawContentBlockStopEvent(index=0, type="content_block_stop"),
            RawMessageDeltaEvent(
                delta=Delta(stop_reason="end_turn", stop_sequence=None),
                type="message_delta",
                usage=MessageDeltaUsage(output_tokens=10),
            ),
            RawMessageStopEvent(type="message_stop"),
        ]


@pytest.mark.asyncio
async def test_anthropic_chat_completion_async(capsys, anthropic_async_client):
    with assert_played("test_anthropic_chat_completion_async.yaml"):
        completion = await anthropic_async_client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is the capital of Spain?"}],
            model="claude-3-haiku-20240307",
        )

    captured = capsys.readouterr()
    assert "WARNING: While" not in captured.err
    assert completion.content is not None


@pytest.mark.asyncio
async def test_anthropic_chat_completion_async_stream_with_pl_id(anthropic_async_client):
    with assert_played("test_anthropic_chat_completion_async_stream_with_pl_id.yaml"):
        completions_gen = await anthropic_async_client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is the capital of Japan?"}],
            model="claude-3-haiku-20240307",
            stream=True,
            return_pl_id=True,
        )

        completions = [completion async for completion, _ in completions_gen]
        assert completions == [
            RawMessageStartEvent(
                message=Message(
                    id="msg_01Bi6S5crUgtL7PUCuYc8Vy6",
                    content=[],
                    model="claude-3-haiku-20240307",
                    role="assistant",
                    stop_reason=None,
                    stop_sequence=None,
                    type="message",
                    usage=Usage(
                        cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=14, output_tokens=4
                    ),
                ),
                type="message_start",
            ),
            RawContentBlockStartEvent(
                content_block=TextBlock(citations=None, text="", type="text"), index=0, type="content_block_start"
            ),
            RawContentBlockDeltaEvent(
                delta=TextDelta(text="The capital of Japan", type="text_delta"), index=0, type="content_block_delta"
            ),
            RawContentBlockDeltaEvent(
                delta=TextDelta(text=" is Tokyo.", type="text_delta"), index=0, type="content_block_delta"
            ),
            RawContentBlockStopEvent(index=0, type="content_block_stop"),
            RawMessageDeltaEvent(
                delta=Delta(stop_reason="end_turn", stop_sequence=None),
                type="message_delta",
                usage=MessageDeltaUsage(output_tokens=10),
            ),
            RawMessageStopEvent(type="message_stop"),
        ]
