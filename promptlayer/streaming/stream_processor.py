"""
Stream processors for handling streaming responses

This module contains the main streaming logic that processes streaming responses
from various LLM providers and builds progressive prompt blueprints.
"""

from typing import Any, AsyncGenerator, AsyncIterable, Callable, Dict, Generator

from .blueprint_builder import (
    build_prompt_blueprint_from_anthropic_event,
    build_prompt_blueprint_from_google_event,
    build_prompt_blueprint_from_openai_chunk,
)


def stream_response(*, generator: Generator, after_stream: Callable, map_results: Callable, metadata: Dict):
    """
    Process streaming responses and build progressive prompt blueprints

    Supports OpenAI, Anthropic, and Google (Gemini) streaming formats, building blueprints
    progressively as the stream progresses.
    """
    results = []
    stream_blueprint = None
    for result in generator:
        results.append(result)

        # Handle OpenAI streaming format - process each chunk individually
        if hasattr(result, "choices"):
            stream_blueprint = build_prompt_blueprint_from_openai_chunk(result, metadata)

        # Handle Google streaming format (Gemini) - GenerateContentResponse objects
        elif hasattr(result, "candidates"):
            stream_blueprint = build_prompt_blueprint_from_google_event(result, metadata)

        # Handle Anthropic streaming format - process each event individually
        elif hasattr(result, "type"):
            stream_blueprint = build_prompt_blueprint_from_anthropic_event(result, metadata)

        data = {
            "request_id": None,
            "raw_response": result,
            "prompt_blueprint": stream_blueprint,
        }
        yield data

    request_response = map_results(results)
    response = after_stream(request_response=request_response.model_dump(mode="json"))
    data["request_id"] = response.get("request_id")
    data["prompt_blueprint"] = response.get("prompt_blueprint")
    yield data


async def astream_response(
    generator: AsyncIterable[Any],
    after_stream: Callable[..., Any],
    map_results: Callable[[Any], Any],
    metadata: Dict[str, Any] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async version of stream_response

    Process streaming responses asynchronously and build progressive prompt blueprints
    Supports OpenAI, Anthropic, and Google (Gemini) streaming formats.
    """
    results = []
    stream_blueprint = None

    async for result in generator:
        results.append(result)

        # Handle OpenAI streaming format - process each chunk individually
        if hasattr(result, "choices"):
            stream_blueprint = build_prompt_blueprint_from_openai_chunk(result, metadata)

        # Handle Google streaming format (Gemini) - GenerateContentResponse objects
        elif hasattr(result, "candidates"):
            stream_blueprint = build_prompt_blueprint_from_google_event(result, metadata)

        # Handle Anthropic streaming format - process each event individually
        elif hasattr(result, "type"):
            stream_blueprint = build_prompt_blueprint_from_anthropic_event(result, metadata)

        data = {
            "request_id": None,
            "raw_response": result,
            "prompt_blueprint": stream_blueprint,
        }
        yield data

    async def async_generator_from_list(lst):
        for item in lst:
            yield item

    request_response = await map_results(async_generator_from_list(results))
    after_stream_response = await after_stream(request_response=request_response.model_dump(mode="json"))
    data["request_id"] = after_stream_response.get("request_id")
    data["prompt_blueprint"] = after_stream_response.get("prompt_blueprint")
    yield data
