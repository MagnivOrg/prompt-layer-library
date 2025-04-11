import os
import time
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails

from tests.utils.vcr import assert_played


@patch("promptlayer.utils.URL_API_PROMPTLAYER", "http://localhost:8000")
@pytest.mark.asyncio
async def test_publish_template_async(sample_template_name, sample_template_content, promptlayer_client):
    body = {
        "prompt_name": sample_template_name,
        "prompt_template": sample_template_content,
        "tags": ["test"],
        "commit_message": "test",
        "metadata": {
            "model": {
                "name": "gpt-4o-mini",
                "provider": "openai",
                "parameters": {
                    "frequency_penalty": 0,
                    "max_tokens": 256,
                    "messages": [{"content": "Hello", "role": "system"}],
                    "model": "gpt-4o",
                    "presence_penalty": 0,
                    "seed": 0,
                    "temperature": 1,
                    "top_p": 1,
                },
            }
        },
    }
    with assert_played("test_publish_template_async.yaml"):
        response = promptlayer_client.templates.publish(body)

    assert response == {
        "id": 4,
        "prompt_name": "sample_template",
        "tags": ["test"],
        "prompt_template": {
            "messages": [
                {
                    "input_variables": [],
                    "template_format": "f-string",
                    "content": [{"type": "text", "text": ""}],
                    "raw_request_display_role": "",
                    "dataset_examples": [],
                    "role": "system",
                    "name": None,
                },
                {
                    "input_variables": [],
                    "template_format": "f-string",
                    "content": [{"type": "text", "text": "What is the capital of Japan?"}],
                    "raw_request_display_role": "",
                    "dataset_examples": [],
                    "role": "user",
                    "name": None,
                },
            ],
            "functions": [],
            "tools": None,
            "function_call": "none",
            "tool_choice": None,
            "type": "chat",
            "input_variables": [],
            "dataset_examples": [],
        },
        "commit_message": "test",
        "metadata": {
            "model": {
                "provider": "openai",
                "name": "gpt-4o-mini",
                "parameters": {
                    "frequency_penalty": 0,
                    "max_tokens": 256,
                    "messages": [{"content": "Hello", "role": "system"}],
                    "model": "gpt-4o",
                    "presence_penalty": 0,
                    "seed": 0,
                    "temperature": 1,
                    "top_p": 1,
                },
            }
        },
        "release_labels": None,
    }


@patch("promptlayer.utils.URL_API_PROMPTLAYER", "http://localhost:8000")
@pytest.mark.asyncio
async def test_get_template_async(sample_template_name, promptlayer_async_client):
    params = {"provider": "openai", "model": "gpt-3.5-turbo"}
    with assert_played("test_get_template_async.yaml"):
        response = await promptlayer_async_client.templates.get(sample_template_name, params)

    assert response == {
        "id": 4,
        "prompt_name": "sample_template",
        "tags": ["test"],
        "workspace_id": 1,
        "commit_message": "test",
        "metadata": {
            "model": {
                "provider": "openai",
                "name": "gpt-4o-mini",
                "parameters": {
                    "frequency_penalty": 0,
                    "max_tokens": 256,
                    "messages": [{"content": "Hello", "role": "system"}],
                    "model": "gpt-4o",
                    "presence_penalty": 0,
                    "seed": 0,
                    "temperature": 1,
                    "top_p": 1,
                },
            }
        },
        "prompt_template": {
            "messages": [
                {
                    "input_variables": [],
                    "template_format": "f-string",
                    "content": [{"type": "text", "text": ""}],
                    "raw_request_display_role": "",
                    "dataset_examples": [],
                    "role": "system",
                    "name": None,
                },
                {
                    "input_variables": [],
                    "template_format": "f-string",
                    "content": [{"type": "text", "text": "What is the capital of Japan?"}],
                    "raw_request_display_role": "",
                    "dataset_examples": [],
                    "role": "user",
                    "name": None,
                },
            ],
            "functions": [],
            "tools": None,
            "function_call": "none",
            "tool_choice": None,
            "type": "chat",
            "input_variables": [],
            "dataset_examples": [],
        },
        "llm_kwargs": {
            "messages": [{"content": "Hello", "role": "system"}],
            "model": "gpt-4o",
            "frequency_penalty": 0,
            "max_tokens": 256,
            "presence_penalty": 0,
            "seed": 0,
            "temperature": 1,
            "top_p": 1,
        },
        "provider_base_url": None,
        "version": 1,
        "snippets": [],
        "warning": None,
    }


@patch("promptlayer.utils.URL_API_PROMPTLAYER", "http://localhost:8000")
@pytest.mark.asyncio
async def test_run_prompt_async(sample_template_name, promptlayer_async_client):
    client = promptlayer_async_client
    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sk-sanitized")}),
        assert_played("test_run_prompt_async.yaml"),
    ):
        response = await client.run(prompt_name=sample_template_name, input_variables={})
        assert response == {
            "request_id": 129,
            "raw_response": ChatCompletion(
                id="chatcmpl-BMAu3fBRyPYaswyFgBBnBQcB0YxUK",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        logprobs=None,
                        message=ChatCompletionMessage(
                            content="Hello! Yes, I'm trained on data up to October 2023. How can I assist you today?",
                            refusal=None,
                            role="assistant",
                            audio=None,
                            function_call=None,
                            tool_calls=None,
                            annotations=[],
                        ),
                    )
                ],
                created=1744624827,
                model="gpt-4o-2024-08-06",
                object="chat.completion",
                service_tier="default",
                system_fingerprint="fp_92f14e8683",
                usage=CompletionUsage(
                    completion_tokens=23,
                    prompt_tokens=8,
                    total_tokens=31,
                    completion_tokens_details=CompletionTokensDetails(
                        accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0
                    ),
                    prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
                ),
            ),
            "prompt_blueprint": {
                "prompt_template": {
                    "messages": [
                        {
                            "input_variables": [],
                            "template_format": "f-string",
                            "content": [{"type": "text", "text": "Hello"}],
                            "raw_request_display_role": "system",
                            "dataset_examples": [],
                            "role": "system",
                            "name": None,
                        },
                        {
                            "input_variables": [],
                            "template_format": "f-string",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Hello! Yes, I'm trained on data up to October 2023. "
                                        "How can I assist you today?"
                                    ),
                                }
                            ],
                            "raw_request_display_role": "assistant",
                            "dataset_examples": [],
                            "role": "assistant",
                            "function_call": None,
                            "name": None,
                            "tool_calls": None,
                        },
                    ],
                    "functions": [],
                    "tools": [],
                    "function_call": None,
                    "tool_choice": None,
                    "type": "chat",
                    "input_variables": [],
                    "dataset_examples": [],
                },
                "commit_message": None,
                "metadata": {
                    "model": {
                        "provider": "openai",
                        "name": "gpt-4o",
                        "parameters": {
                            "frequency_penalty": 0,
                            "max_tokens": 256,
                            "presence_penalty": 0,
                            "seed": 0,
                            "temperature": 1,
                            "top_p": 1,
                            "stream": False,
                        },
                    }
                },
                "provider_base_url_name": None,
                "report_id": None,
                "inference_client_name": None,
            },
        }

        if (pl_request_id := response["request_id"]) is not None:
            return

        await client.track.metadata(pl_request_id, {"test": "test"})
        await client.track.score(pl_request_id, 100)
        await client.track.prompt(pl_request_id, sample_template_name, {})


@patch("promptlayer.utils.URL_API_PROMPTLAYER", "http://localhost:8000")
@pytest.mark.asyncio
async def test_log_request_async(sample_template_name, promptlayer_async_client):
    with assert_played("test_log_request_async.yaml"):
        template = await promptlayer_async_client.templates.get(sample_template_name)
        assert template

        await promptlayer_async_client.log_request(
            provider="openai",
            model="gpt-4-mini",
            input=template["prompt_template"],
            output=template["prompt_template"],
            request_start_time=time.time(),
            request_end_time=time.time(),
        )
