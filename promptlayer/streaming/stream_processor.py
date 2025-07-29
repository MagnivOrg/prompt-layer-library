from typing import Any, AsyncGenerator, AsyncIterable, Callable, Dict, Generator

from .blueprint_builder import (
    build_prompt_blueprint_from_anthropic_event,
    build_prompt_blueprint_from_google_event,
    build_prompt_blueprint_from_openai_chunk,
)


def _build_stream_blueprint(result: Any, metadata: Dict) -> Any:
    model_info = metadata.get("model", {}) if metadata else {}
    provider = model_info.get("provider", "")
    model_name = model_info.get("name", "")

    if provider == "openai" or provider == "openai.azure":
        return build_prompt_blueprint_from_openai_chunk(result, metadata)

    elif provider == "google" or (provider == "vertexai" and model_name.startswith("gemini")):
        return build_prompt_blueprint_from_google_event(result, metadata)

    elif provider == "anthropic" or (provider == "vertexai" and model_name.startswith("claude")):
        return build_prompt_blueprint_from_anthropic_event(result, metadata)

    elif provider == "mistral":
        return build_prompt_blueprint_from_openai_chunk(result.data, metadata)

    return None


def _build_stream_data(result: Any, stream_blueprint: Any, request_id: Any = None) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "raw_response": result,
        "prompt_blueprint": stream_blueprint,
    }


def stream_response(*, generator: Generator, after_stream: Callable, map_results: Callable, metadata: Dict):
    results = []
    for result in generator:
        results.append(result)

        stream_blueprint = _build_stream_blueprint(result, metadata)
        data = _build_stream_data(result, stream_blueprint)
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
    results = []

    async for result in generator:
        results.append(result)

        stream_blueprint = _build_stream_blueprint(result, metadata)
        data = _build_stream_data(result, stream_blueprint)
        yield data

    async def async_generator_from_list(lst):
        for item in lst:
            yield item

    request_response = await map_results(async_generator_from_list(results))
    after_stream_response = await after_stream(request_response=request_response.model_dump(mode="json"))
    data["request_id"] = after_stream_response.get("request_id")
    data["prompt_blueprint"] = after_stream_response.get("prompt_blueprint")
    yield data
