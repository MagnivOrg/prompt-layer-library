import datetime
import os
from copy import deepcopy
from functools import wraps
from typing import Any, Dict, List, Literal, Sequence, Union

import requests
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.semconv.resource import ResourceAttributes

from promptlayer.groups import GroupManager
from promptlayer.promptlayer import PromptLayerBase
from promptlayer.templates import TemplateManager
from promptlayer.track import TrackManager
from promptlayer.types.prompt_template import GetPromptTemplate
from promptlayer.utils import (
    URL_API_PROMPTLAYER,
    anthropic_request,
    anthropic_stream_completion,
    anthropic_stream_message,
    openai_request,
    openai_stream_chat,
    openai_stream_completion,
    stream_response,
    track_request,
)


class PromptLayerSpanExporter(SpanExporter):
    def __init__(self, api_key=None):
        self.url = f"{URL_API_PROMPTLAYER}/spans-bulk"
        self.api_key = api_key

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        request_data = []

        for span in spans:
            span_info = {
                "name": span.name,
                "context": {
                    "trace_id": hex(span.context.trace_id)[2:].zfill(
                        32
                    ),  # Ensure 32 characters
                    "span_id": hex(span.context.span_id)[2:].zfill(
                        16
                    ),  # Ensure 16 characters
                    "trace_state": str(span.context.trace_state),
                },
                "kind": str(span.kind),
                "parent_id": hex(span.parent.span_id)[2:] if span.parent else None,
                "start_time": span.start_time,
                "end_time": span.end_time,
                "status": {
                    "status_code": str(span.status.status_code),
                    "description": span.status.description,
                },
                "attributes": dict(span.attributes),
                "events": [
                    {
                        "name": event.name,
                        "timestamp": event.timestamp,
                        "attributes": dict(event.attributes),
                    }
                    for event in span.events
                ],
                "links": [
                    {"context": link.context, "attributes": dict(link.attributes)}
                    for link in span.links
                ],
                "resource": {
                    "attributes": dict(span.resource.attributes),
                    "schema_url": span.resource.schema_url,
                },
            }
            request_data.append(span_info)

        try:
            response = requests.post(
                self.url,
                headers={
                    "X-Api-Key": self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "spans": request_data,
                    "workspace_id": 1,
                },
            )
            response.raise_for_status()
            return SpanExportResult.SUCCESS
        except requests.RequestException:
            return SpanExportResult.FAILURE

    def shutdown(self):
        pass


MAP_PROVIDER_TO_FUNCTION_NAME = {
    "openai": {
        "chat": {
            "function_name": "openai.chat.completions.create",
            "stream_function": openai_stream_chat,
        },
        "completion": {
            "function_name": "openai.completions.create",
            "stream_function": openai_stream_completion,
        },
    },
    "anthropic": {
        "chat": {
            "function_name": "anthropic.messages.create",
            "stream_function": anthropic_stream_message,
        },
        "completion": {
            "function_name": "anthropic.completions.create",
            "stream_function": anthropic_stream_completion,
        },
    },
}

MAP_PROVIDER_TO_FUNCTION = {
    "openai": openai_request,
    "anthropic": anthropic_request,
}


class PromptLayer:
    def __init__(self, api_key: str = None, enable_tracing: bool = False):
        if api_key is None:
            api_key = os.environ.get("PROMPTLAYER_API_KEY")
        if api_key is None:
            raise ValueError(
                "PromptLayer API key not provided. Please set the PROMPTLAYER_API_KEY environment variable or pass the api_key parameter."
            )
        self.api_key = api_key
        self.templates = TemplateManager(api_key)
        self.group = GroupManager(api_key)
        self.track = TrackManager(api_key)

        if enable_tracing:
            resource = Resource(
                attributes={ResourceAttributes.SERVICE_NAME: "prompt-layer-library"}
            )
            tracer_provider = TracerProvider(resource=resource)
            promptlayer_exporter = PromptLayerSpanExporter(api_key=self.api_key)
            span_processor = BatchSpanProcessor(promptlayer_exporter)
            tracer_provider.add_span_processor(span_processor)
            trace.set_tracer_provider(tracer_provider)
            self.tracer = trace.get_tracer(__name__)
        else:
            self.tracer = None

    def __getattr__(
        self,
        name: Union[Literal["openai"], Literal["anthropic"], Literal["prompts"]],
    ):
        if name == "openai":
            import openai as openai_module

            openai = PromptLayerBase(
                openai_module, function_name="openai", api_key=self.api_key
            )
            return openai
        elif name == "anthropic":
            import anthropic as anthropic_module

            anthropic = PromptLayerBase(
                anthropic_module,
                function_name="anthropic",
                provider_type="anthropic",
                api_key=self.api_key,
            )
            return anthropic
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")

    def run(
        self,
        prompt_name: str,
        prompt_version: Union[int, None] = None,
        prompt_release_label: Union[str, None] = None,
        input_variables: Union[Dict[str, Any], None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, str], None] = None,
        group_id: Union[int, None] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        if self.tracer:
            with self.tracer.start_as_current_span("PromptLayer.run") as main_span:
                main_span.set_attribute("prompt_name", prompt_name)
                main_span.set_attribute("stream", stream)
                return self._run_internal(
                    prompt_name,
                    prompt_version,
                    prompt_release_label,
                    input_variables,
                    tags,
                    metadata,
                    group_id,
                    stream,
                )
        else:
            return self._run_internal(
                prompt_name,
                prompt_version,
                prompt_release_label,
                input_variables,
                tags,
                metadata,
                group_id,
                stream,
            )

    def _run_internal(
        self,
        prompt_name: str,
        prompt_version: Union[int, None] = None,
        prompt_release_label: Union[str, None] = None,
        input_variables: Union[Dict[str, Any], None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, str], None] = None,
        group_id: Union[int, None] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        # Prepare parameters for getting prompt template
        template_get_params: GetPromptTemplate = {}
        if prompt_version:
            template_get_params["version"] = prompt_version
        if prompt_release_label:
            template_get_params["label"] = prompt_release_label
        if input_variables:
            template_get_params["input_variables"] = input_variables
        if metadata:
            template_get_params["metadata_filters"] = metadata

        # Get prompt blueprint
        if self.tracer:
            with self.tracer.start_as_current_span(
                "fetch_prompt_template"
            ) as fetch_prompt_template_span:
                fetch_prompt_template_span.set_attribute("prompt_name", prompt_name)
                prompt_blueprint = self.templates.get(prompt_name, template_get_params)
        else:
            prompt_blueprint = self.templates.get(prompt_name, template_get_params)

        prompt_template = prompt_blueprint["prompt_template"]

        # Validate prompt blueprint
        if not prompt_blueprint["llm_kwargs"]:
            raise ValueError(
                f"Prompt '{prompt_name}' does not have any LLM kwargs associated with it."
            )

        prompt_blueprint_metadata = prompt_blueprint.get("metadata")
        if not prompt_blueprint_metadata:
            raise ValueError(
                f"Prompt '{prompt_name}' does not have any metadata associated with it."
            )

        prompt_blueprint_model = prompt_blueprint_metadata.get("model")
        if not prompt_blueprint_model:
            raise ValueError(
                f"Prompt '{prompt_name}' does not have a model parameters associated with it."
            )

        # Prepare request parameters
        provider = prompt_blueprint_model["provider"]
        request_start_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
        kwargs = deepcopy(prompt_blueprint["llm_kwargs"])
        config = MAP_PROVIDER_TO_FUNCTION_NAME[provider][prompt_template["type"]]
        function_name = config["function_name"]
        stream_function = config["stream_function"]
        request_function = MAP_PROVIDER_TO_FUNCTION[provider]

        # Set provider base URL if available
        provider_base_url = prompt_blueprint.get("provider_base_url")
        if provider_base_url:
            kwargs["base_url"] = provider_base_url["url"]

        # Set streaming options
        kwargs["stream"] = stream
        if stream and provider == "openai":
            kwargs["stream_options"] = {"include_usage": True}

        # Make the request
        if self.tracer:
            with self.tracer.start_as_current_span("llm_request") as llm_request_span:
                llm_request_span.set_attribute("provider", provider)
                llm_request_span.set_attribute("function_name", function_name)
                response = request_function(prompt_blueprint, **kwargs)

                # Define tracking function
                def _track_request(**body):
                    request_end_time = datetime.datetime.now(
                        datetime.timezone.utc
                    ).timestamp()

                    with self.tracer.start_as_current_span(
                        "track_request"
                    ) as track_request_span:
                        track_request_span.set_attribute("function_name", function_name)
                        track_request_span.set_attribute("provider_type", provider)

                        request_log = track_request(
                            function_name=function_name,
                            provider_type=provider,
                            args=[],
                            kwargs=kwargs,
                            tags=tags,
                            request_start_time=request_start_time,
                            request_end_time=request_end_time,
                            api_key=self.api_key,
                            metadata=metadata,
                            prompt_id=prompt_blueprint["id"],
                            prompt_version=prompt_blueprint["version"],
                            prompt_input_variables=input_variables,
                            group_id=group_id,
                            return_prompt_blueprint=True,
                            **body,
                        )

                        llm_request_span.set_attribute(
                            "request_log_id", request_log["request_id"]
                        )
                        return request_log

                # Handle streaming response
                if stream:
                    return stream_response(response, _track_request, stream_function)

                # Handle non-streaming response
                request_log = _track_request(request_response=response.model_dump())

                return {
                    "request_id": request_log["request_id"],
                    "raw_response": response,
                    "prompt_blueprint": request_log["prompt_blueprint"],
                }
        else:
            response = request_function(prompt_blueprint, **kwargs)

            # Define tracking function
            def _track_request(**body):
                request_end_time = datetime.datetime.now(
                    datetime.timezone.utc
                ).timestamp()
                return track_request(
                    function_name=function_name,
                    provider_type=provider,
                    args=[],
                    kwargs=kwargs,
                    tags=tags,
                    request_start_time=request_start_time,
                    request_end_time=request_end_time,
                    api_key=self.api_key,
                    metadata=metadata,
                    prompt_id=prompt_blueprint["id"],
                    prompt_version=prompt_blueprint["version"],
                    prompt_input_variables=input_variables,
                    group_id=group_id,
                    return_prompt_blueprint=True,
                    **body,
                )

            # Handle streaming response
            if stream:
                return stream_response(response, _track_request, stream_function)

            # Handle non-streaming response
            request_log = _track_request(request_response=response.model_dump())

            return {
                "request_id": request_log["request_id"],
                "raw_response": response,
                "prompt_blueprint": request_log["prompt_blueprint"],
            }

    def traceable(self, run_type=None, metadata=None):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.tracer:
                    with self.tracer.start_as_current_span(func.__name__) as span:
                        if run_type:
                            span.set_attribute("run_type", run_type)

                        if metadata:
                            for key, value in metadata.items():
                                span.set_attribute(key, value)

                        promptlayer_extra = kwargs.pop("promptlayer_extra", {})
                        run_id = promptlayer_extra.get("run_id")

                        if run_id:
                            span.set_attribute("run_id", run_id)

                        extra_metadata = promptlayer_extra.get("metadata", {})

                        for key, value in extra_metadata.items():
                            span.set_attribute(key, value)

                        result = func(*args, **kwargs)

                        return result
                else:
                    return func(*args, **kwargs)

            return wrapper

        return decorator


__version__ = "1.0.9"
__all__ = ["PromptLayer", "__version__"]
