from typing import Any, AsyncGenerator, AsyncIterable, Callable, Dict, Generator

from .blueprint_builder import (
    build_prompt_blueprint_from_anthropic_event,
    build_prompt_blueprint_from_bedrock_event,
    build_prompt_blueprint_from_google_event,
    build_prompt_blueprint_from_openai_chunk,
    build_prompt_blueprint_from_openai_responses_event,
)


def _build_stream_blueprint(result: Any, metadata: Dict) -> Any:
    model_info = metadata.get("model", {}) if metadata else {}
    provider = model_info.get("provider", "")
    model_name = model_info.get("name", "")

    if provider == "openai" or provider == "openai.azure":
        api_type = model_info.get("api_type", "chat-completions") if metadata else "chat-completions"
        if api_type == "chat-completions":
            return build_prompt_blueprint_from_openai_chunk(result, metadata)
        elif api_type == "responses":
            return build_prompt_blueprint_from_openai_responses_event(result, metadata)

    elif provider == "google" or (provider == "vertexai" and model_name.startswith("gemini")):
        return build_prompt_blueprint_from_google_event(result, metadata)

    elif provider in ("anthropic", "anthropic.bedrock") or (provider == "vertexai" and model_name.startswith("claude")):
        return build_prompt_blueprint_from_anthropic_event(result, metadata)

    elif provider == "mistral":
        return build_prompt_blueprint_from_openai_chunk(result.data, metadata)

    elif provider == "amazon.bedrock":
        return build_prompt_blueprint_from_bedrock_event(result, metadata)

    return None


def _build_stream_data(result: Any, stream_blueprint: Any, request_id: Any = None) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "raw_response": result,
        "prompt_blueprint": stream_blueprint,
    }


def stream_response(*, generator: Generator, after_stream: Callable, map_results: Callable, metadata: Dict):
    results = []
    provider = metadata.get("model", {}).get("provider", "")
    if provider == "amazon.bedrock":
        response_metadata = generator.get("ResponseMetadata", {})
        generator = generator.get("stream", generator)

    for result in generator:
        results.append(result)

        stream_blueprint = _build_stream_blueprint(result, metadata)
        data = _build_stream_data(result, stream_blueprint)
        yield data

    request_response = map_results(results)
    if provider == "amazon.bedrock":
        request_response["ResponseMetadata"] = response_metadata
    else:
        request_response = request_response.model_dump(mode="json")

    response = after_stream(request_response=request_response)
    data["request_id"] = response.get("request_id")
    data["prompt_blueprint"] = response.get("prompt_blueprint")
    yield data


async def astream_response(
    generator: AsyncIterable[Any],
    after_stream: Callable[..., Any],
    map_results: Callable[[Any], Any],
    metadata: Dict[str, Any] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    results = []
    provider = metadata.get("model", {}).get("provider", "")
    if provider == "amazon.bedrock":
        response_metadata = generator.get("ResponseMetadata", {})
        generator = generator.get("stream", generator)

    async for result in generator:
        results.append(result)

        stream_blueprint = _build_stream_blueprint(result, metadata)
        data = _build_stream_data(result, stream_blueprint)
        yield data

    async def async_generator_from_list(lst):
        for item in lst:
            yield item

    request_response = await map_results(async_generator_from_list(results))

    if provider == "amazon.bedrock":
        request_response["ResponseMetadata"] = response_metadata
    else:
        request_response = request_response.model_dump(mode="json")

    after_stream_response = await after_stream(request_response=request_response)
    data["request_id"] = after_stream_response.get("request_id")
    data["prompt_blueprint"] = after_stream_response.get("prompt_blueprint")
    yield data
