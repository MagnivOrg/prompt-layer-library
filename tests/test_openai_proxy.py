import os

import pytest

from promptlayer import PromptLayer


def test_openai_chat_completion(capsys):
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    OpenAI = promptlayer.openai.OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of the United States?"},
        ],
    )
    captured = capsys.readouterr()
    assert "WARNING: While" not in captured.err
    assert completion.choices[0].message.content is not None


def test_openai_chat_completion_with_pl_id(capsys):
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    OpenAI = promptlayer.openai.OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    completion, pl_id = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        return_pl_id=True,
    )
    captured = capsys.readouterr()
    assert "WARNING: While" not in captured.err
    assert completion.choices[0].message.content is not None
    assert isinstance(pl_id, int)


def test_openai_chat_completion_with_stream(capsys):
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    OpenAI = promptlayer.openai.OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Germany?"},
        ],
        stream=True,
    )
    captured = capsys.readouterr()
    for chunk in completion:
        assert chunk.choices[0].delta != {}
    assert "WARNING: While" not in captured.err


def test_openai_chat_completion_with_stream_and_pl_id(capsys):
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    OpenAI = promptlayer.openai.OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Italy?"},
        ],
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
async def test_openai_chat_completion_async(capsys):
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    AsyncOpenAI = promptlayer.openai.AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Spain?"},
        ],
    )
    captured = capsys.readouterr()
    assert "WARNING: While" not in captured.err
    assert completion.choices[0].message.content is not None


@pytest.mark.asyncio
async def test_openai_chat_completion_async_stream_with_pl_id(capsys):
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    AsyncOpenAI = promptlayer.openai.AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Japan?"},
        ],
        stream=True,
        return_pl_id=True,
    )
    captured = capsys.readouterr()
    assert "WARNING: While" not in captured.err
    pl_id = None
    async for _, pl_id in completion:
        assert pl_id is None or isinstance(pl_id, int)
    assert isinstance(pl_id, int)
