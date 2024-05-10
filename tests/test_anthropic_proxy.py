import os

import pytest

from promptlayer import PromptLayer


def test_anthropic_chat_completion(capsys):
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    Anthropic = promptlayer.anthropic.Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    completion = client.messages.create(
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What is the capital of the United States?"},
        ],
        model="claude-3-haiku-20240307",
    )
    captured = capsys.readouterr()
    assert "WARNING: While" not in captured.err
    assert completion.content is not None


def test_anthropic_chat_completion_with_pl_id(capsys):
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    Anthropic = promptlayer.anthropic.Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    completion, pl_id = client.messages.create(
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        model="claude-3-haiku-20240307",
        return_pl_id=True,
    )
    captured = capsys.readouterr()
    assert "WARNING: While" not in captured.err
    assert completion.content is not None
    assert isinstance(pl_id, int)


def test_anthropic_chat_completion_with_stream(capsys):
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    Anthropic = promptlayer.anthropic.Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    completion = client.messages.create(
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What is the capital of Germany?"},
        ],
        model="claude-3-haiku-20240307",
        stream=True,
    )
    captured = capsys.readouterr()
    for chunk in completion:
        pass
    assert "WARNING: While" not in captured.err


def test_anthropic_chat_completion_with_stream_and_pl_id(capsys):
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    Anthropic = promptlayer.anthropic.Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    completion = client.messages.create(
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What is the capital of Italy?"},
        ],
        model="claude-3-haiku-20240307",
        stream=True,
        return_pl_id=True,
    )
    captured = capsys.readouterr()
    pl_id = None
    for _, pl_id in completion:
        assert pl_id is None or isinstance(pl_id, int)
    assert isinstance(pl_id, int)
    assert "WARNING: While" not in captured.err


@pytest.mark.asyncio
async def test_anthropic_chat_completion_async(capsys):
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    AsyncAnthropic = promptlayer.anthropic.AsyncAnthropic
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    completion = await client.messages.create(
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What is the capital of Spain?"},
        ],
        model="claude-3-haiku-20240307",
    )
    captured = capsys.readouterr()
    assert "WARNING: While" not in captured.err
    assert completion.content is not None


@pytest.mark.asyncio
async def test_anthropic_chat_completion_async_stream_with_pl_id(capsys):
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    AsyncAnthropic = promptlayer.anthropic.AsyncAnthropic
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    completion = await client.messages.create(
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What is the capital of Japan?"},
        ],
        model="claude-3-haiku-20240307",
        stream=True,
        return_pl_id=True,
    )
    captured = capsys.readouterr()
    pl_id = None
    async for _, pl_id in completion:
        assert pl_id is None or isinstance(pl_id, int)
    assert isinstance(pl_id, int)
    assert "WARNING: While" not in captured.err
