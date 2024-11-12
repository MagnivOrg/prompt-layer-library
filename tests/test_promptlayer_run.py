# tests/test_async_promptlayer.py

import os
import time

import pytest

from promptlayer import AsyncPromptLayer, PromptLayer

template_name = f"test_template_{int(time.time())}"


@pytest.mark.asyncio
async def test_async_template_publish_and_get():
    """
    Test publishing and retrieving a template asynchronously.
    """
    async_pl = AsyncPromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    pl = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    template_content = {
        "dataset_examples": [],
        "function_call": "none",
        "functions": [],
        "input_variables": [],
        "messages": [
            {
                "content": [{"text": "", "type": "text"}],
                "dataset_examples": [],
                "input_variables": [],
                "name": None,
                "raw_request_display_role": "",
                "role": "system",
                "template_format": "f-string",
            },
            {
                "content": [{"text": "What is the capital of Japan?", "type": "text"}],
                "dataset_examples": [],
                "input_variables": [],
                "name": None,
                "raw_request_display_role": "",
                "role": "user",
                "template_format": "f-string",
            },
        ],
        "tool_choice": None,
        "tools": None,
        "type": "chat",
    }

    # Publish template
    publish_start = time.time()
    pl.templates.publish(
        {
            "prompt_name": template_name,
            "prompt_template": template_content,
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
    )
    publish_end = time.time()
    print(f"Template publish latency: {publish_end - publish_start}")

    # Retrieve template
    get_start = time.time()
    get_response = await async_pl.templates.get(
        template_name, {"provider": "openai", "model": "gpt-3.5-turbo"}
    )
    get_end = time.time()
    print(f"Template get latency: {get_end - get_start}")

    # Assert that retrieved template matches what was published
    assert get_response["prompt_name"] == template_name
    assert get_response["prompt_template"] == template_content


@pytest.mark.asyncio
async def test_async_run_method():
    """
    Test running a template and tracking associated metadata, score, and prompts.
    """
    async_pl = AsyncPromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    # template_name = "Test"  # Assumes 'Test' template exists

    # Retrieve template details
    template = await async_pl.templates.get(
        template_name, {"provider": "openai", "model": "gpt-3.5-turbo"}
    )
    assert template is not None, f"Template '{template_name}' could not be found."

    # Run template
    run_start = time.time()
    response = await async_pl.run(prompt_name=template_name, input_variables={})
    run_end = time.time()
    print(f"Run method latency: {run_end - run_start}")

    assert response["raw_response"].choices[0].message.content is not None

    pl_request_id = response["request_id"]

    if pl_request_id is None:
        # Track metadata
        track_metadata_start = time.time()
        await async_pl.track.metadata(pl_request_id, {"test": "test"})
        track_metadata_end = time.time()
        print(f"Track metadata latency: {track_metadata_end - track_metadata_start}")

        # Track score
        track_score_start = time.time()
        await async_pl.track.score(pl_request_id, 100)
        track_score_end = time.time()
        print(f"Track score latency: {track_score_end - track_score_start}")

        # Track prompt
        track_prompt_start = time.time()
        await async_pl.track.prompt(pl_request_id, template_name, {})
        track_prompt_end = time.time()
        print(f"Track prompt latency: {track_prompt_end - track_prompt_start}")


@pytest.mark.asyncio
async def test_async_log_request():
    """
    Test logging a request asynchronously.
    """
    async_pl = AsyncPromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    # template_name = "Test"  # Assumes 'Test' template exists

    # Retrieve template details
    template = await async_pl.templates.get(template_name)
    assert template is not None, f"Template '{template_name}' could not be found."

    # Log request
    log_request_start = time.time()
    await async_pl.log_request(
        provider="openai",
        model="gpt-4-mini",
        input=template["prompt_template"],
        output=template["prompt_template"],
        request_start_time=time.time(),
        request_end_time=time.time(),
    )
    log_request_end = time.time()
    print(f"Log request latency: {log_request_end - log_request_start}")
