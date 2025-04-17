import pytest

from promptlayer import AsyncPromptLayer, PromptLayer


@pytest.fixture
def promptlayer_client(promptlayer_api_key):
    return PromptLayer(api_key=promptlayer_api_key)


@pytest.fixture
def promptlayer_async_client(promptlayer_api_key):
    return AsyncPromptLayer(api_key=promptlayer_api_key)


@pytest.fixture
def anthropic_client(promptlayer_client, anthropic_api_key):
    return promptlayer_client.anthropic.Anthropic(api_key=anthropic_api_key)


@pytest.fixture
def anthropic_async_client(promptlayer_client, anthropic_api_key):
    return promptlayer_client.anthropic.AsyncAnthropic(api_key=anthropic_api_key)


@pytest.fixture
def openai_client(promptlayer_client, openai_api_key):
    return promptlayer_client.openai.OpenAI(api_key=openai_api_key)


@pytest.fixture
def openai_async_client(promptlayer_client, openai_api_key):
    return promptlayer_client.openai.AsyncOpenAI(api_key=openai_api_key)
