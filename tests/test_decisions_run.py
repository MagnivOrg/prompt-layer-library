from unittest.mock import AsyncMock, MagicMock

import httpx

import pytest

from promptlayer.promptlayer_mixins import PromptLayerMixin
from promptlayer.track.error_tracking import categorize_error
from promptlayer.utils import aopenai_request, aopenrouter_request, openai_request, openrouter_request

DECISION_KWARGS = {
    "model": "gpt-6-luna",
    "input": "The package arrived with a broken screen.",
    "questions": [{"type": "predicate", "name": "damaged", "instructions": "Is the item damaged?"}],
}
BLUEPRINT = {
    "llm_kwargs": DECISION_KWARGS,
    "metadata": {"model": {"provider": "openai", "name": "gpt-6-luna", "api_type": "decisions"}},
    "prompt_template": {"type": "chat"},
}
DECISION_RESPONSE = {"answers": [{"type": "predicate", "name": "damaged", "probability": 0.95}]}


@pytest.mark.parametrize("is_async", [False, True])
def test_prepare_llm_data_for_decisions(is_async):
    llm_data = PromptLayerMixin._prepare_llm_data(
        prompt_blueprint=BLUEPRINT,
        prompt_template=BLUEPRINT["prompt_template"],
        prompt_blueprint_model=BLUEPRINT["metadata"]["model"],
        stream=False,
        is_async=is_async,
    )
    assert llm_data["function_name"] == "openai.decisions.create"
    assert llm_data["function_kwargs"] == DECISION_KWARGS


def test_decisions_reject_streaming():
    with pytest.raises(ValueError, match="does not support streaming"):
        PromptLayerMixin._prepare_llm_data(
            prompt_blueprint=BLUEPRINT,
            prompt_template=BLUEPRINT["prompt_template"],
            prompt_blueprint_model=BLUEPRINT["metadata"]["model"],
            stream=True,
        )


def test_openai_request_uses_decisions_resource(monkeypatch):
    client = MagicMock()
    client.decisions.create.return_value = DECISION_RESPONSE
    monkeypatch.setattr("promptlayer.utils._get_cached_client", lambda key, factory: client)

    assert openai_request(BLUEPRINT, {}, DECISION_KWARGS) == DECISION_RESPONSE
    client.decisions.create.assert_called_once_with(**DECISION_KWARGS)


def test_openai_request_falls_back_to_raw_post(monkeypatch):
    client = MagicMock(spec=["post"])
    client.post.return_value.json.return_value = DECISION_RESPONSE
    monkeypatch.setattr("promptlayer.utils._get_cached_client", lambda key, factory: client)

    assert openai_request(BLUEPRINT, {}, DECISION_KWARGS) == DECISION_RESPONSE
    assert client.post.call_args.args == ("/decisions",)
    assert client.post.call_args.kwargs["body"] == DECISION_KWARGS


@pytest.mark.asyncio
async def test_aopenai_request_uses_decisions_resource(monkeypatch):
    client = MagicMock()
    client.decisions.create = AsyncMock(return_value=DECISION_RESPONSE)
    monkeypatch.setattr("promptlayer.utils._get_cached_client", lambda key, factory: client)

    assert await aopenai_request(BLUEPRINT, {}, DECISION_KWARGS) == DECISION_RESPONSE


OPENROUTER_DECISION_KWARGS = {
    "model": "openai/gpt-6-luna-decisions",
    "state": "The package arrived with a broken screen.",
    "questions": {"damaged": {"type": "noul", "instructions": "Is the item damaged?"}},
}
OPENROUTER_BLUEPRINT = {
    "llm_kwargs": OPENROUTER_DECISION_KWARGS,
    "metadata": {"model": {"provider": "openrouter", "name": "openai/gpt-6-luna-decisions", "api_type": "decisions"}},
    "prompt_template": {"type": "chat"},
}
OPENROUTER_DECISION_RESPONSE = {
    "id": "dec_123",
    "model": "openai/gpt-6-luna-decisions",
    "provider": "OpenAI",
    "answers": {"damaged": {"type": "noul", "noul": 0.97}},
    "usage": {"input_tokens": 20, "output_tokens": 1, "cost": 0.0001},
}


def _mock_response(json_body, status_code=200, text=""):
    response = MagicMock(status_code=status_code, is_error=status_code >= 400, text=text)
    response.json.return_value = json_body
    return response


@pytest.mark.parametrize("is_async", [False, True])
def test_prepare_llm_data_for_openrouter_decisions(is_async):
    llm_data = PromptLayerMixin._prepare_llm_data(
        prompt_blueprint=OPENROUTER_BLUEPRINT,
        prompt_template=OPENROUTER_BLUEPRINT["prompt_template"],
        prompt_blueprint_model=OPENROUTER_BLUEPRINT["metadata"]["model"],
        stream=False,
        is_async=is_async,
    )
    assert llm_data["provider"] == "openrouter"
    assert llm_data["function_name"] == "openrouter.decisions.create"
    assert llm_data["stream_function"] is None
    assert llm_data["request_function"] is (aopenrouter_request if is_async else openrouter_request)
    assert llm_data["function_kwargs"] == OPENROUTER_DECISION_KWARGS


def test_openrouter_decisions_reject_streaming():
    with pytest.raises(ValueError, match="does not support streaming"):
        PromptLayerMixin._prepare_llm_data(
            prompt_blueprint=OPENROUTER_BLUEPRINT,
            prompt_template=OPENROUTER_BLUEPRINT["prompt_template"],
            prompt_blueprint_model=OPENROUTER_BLUEPRINT["metadata"]["model"],
            stream=True,
        )


def test_openrouter_request_posts_decisions(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    post = MagicMock(return_value=_mock_response(OPENROUTER_DECISION_RESPONSE))
    monkeypatch.setattr("promptlayer.utils.httpx.post", post)
    monkeypatch.setattr("promptlayer.utils._get_openrouter_client", MagicMock(side_effect=AssertionError))

    result = openrouter_request(OPENROUTER_BLUEPRINT, {"api_key": "sk-or-test"}, dict(OPENROUTER_DECISION_KWARGS))

    assert result == OPENROUTER_DECISION_RESPONSE
    kwargs = post.call_args.kwargs
    assert kwargs["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert kwargs["json"] == OPENROUTER_DECISION_KWARGS
    assert kwargs["headers"] == {"Authorization": "Bearer sk-or-test"}


def test_openrouter_decisions_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    post = MagicMock(return_value=_mock_response(OPENROUTER_DECISION_RESPONSE))
    monkeypatch.setattr("promptlayer.utils.httpx.post", post)

    openrouter_request(OPENROUTER_BLUEPRINT, {}, dict(OPENROUTER_DECISION_KWARGS))

    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer sk-or-env"}


@pytest.mark.asyncio
async def test_aopenrouter_request_posts_decisions(monkeypatch):
    client = MagicMock()
    client.post = AsyncMock(return_value=_mock_response(OPENROUTER_DECISION_RESPONSE))
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client)
    client_cm.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("promptlayer.utils.httpx.AsyncClient", lambda: client_cm)

    result = await aopenrouter_request(
        OPENROUTER_BLUEPRINT, {"api_key": "sk-or-test"}, dict(OPENROUTER_DECISION_KWARGS)
    )

    assert result == OPENROUTER_DECISION_RESPONSE
    kwargs = client.post.call_args.kwargs
    assert kwargs["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert kwargs["json"] == OPENROUTER_DECISION_KWARGS
    assert kwargs["headers"] == {"Authorization": "Bearer sk-or-test"}


@pytest.mark.parametrize(
    "status_code, category",
    [(401, "PROVIDER_AUTH_ERROR"), (402, "PROVIDER_QUOTA_LIMIT"), (429, "PROVIDER_RATE_LIMIT")],
)
def test_openrouter_decisions_errors_keep_status_and_body(monkeypatch, status_code, category):
    response = _mock_response(None, status_code=status_code, text="provider says no")
    monkeypatch.setattr("promptlayer.utils.httpx.post", MagicMock(return_value=response))

    with pytest.raises(Exception) as error:
        openrouter_request(OPENROUTER_BLUEPRINT, {"api_key": "sk-or-test"}, dict(OPENROUTER_DECISION_KWARGS))

    assert "provider says no" in str(error.value)
    assert categorize_error(error.value) == category


def test_openrouter_decisions_timeout_is_categorized(monkeypatch):
    monkeypatch.setattr("promptlayer.utils.httpx.post", MagicMock(side_effect=httpx.ReadTimeout("slow")))

    with pytest.raises(TimeoutError) as error:
        openrouter_request(OPENROUTER_BLUEPRINT, {"api_key": "sk-or-test"}, dict(OPENROUTER_DECISION_KWARGS))

    assert categorize_error(error.value) == "PROVIDER_TIMEOUT"


def test_openrouter_decisions_require_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    post = MagicMock()
    monkeypatch.setattr("promptlayer.utils.httpx.post", post)

    with pytest.raises(ValueError, match="OpenRouter API key missing"):
        openrouter_request(OPENROUTER_BLUEPRINT, {}, dict(OPENROUTER_DECISION_KWARGS))
    post.assert_not_called()
