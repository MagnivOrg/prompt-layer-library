import os

import pytest


@pytest.fixture
def promptlayer_api_key():
    return os.environ.get("PROMPTLAYER_API_KEY", "pl_sanitized")


@pytest.fixture
def anthropic_api_key():
    return os.environ.get("ANTHROPIC_API_KEY", "sk-ant-api03-sanitized")


@pytest.fixture
def openai_api_key():
    return os.environ.get("OPENAI_API_KEY", "sk-sanitized")


@pytest.fixture
def base_url():
    return "http://localhost:8000"


@pytest.fixture
def headers(promptlayer_api_key):
    return {"X-API-KEY": promptlayer_api_key}
