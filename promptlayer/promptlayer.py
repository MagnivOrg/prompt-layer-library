import datetime
import os
from copy import deepcopy
from functools import wraps
from typing import Any, Dict, List, Literal, Union

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from promptlayer.groups import GroupManager
from promptlayer.promptlayer_base import PromptLayerBase
from promptlayer.span_exporter import PromptLayerSpanExporter
from promptlayer.templates import TemplateManager
from promptlayer.track import TrackManager
from promptlayer.types.prompt_template import GetPromptTemplate
from promptlayer.utils import (
    anthropic_request,
    anthropic_stream_completion,
    anthropic_stream_message,
    openai_request,
    openai_stream_chat,
    openai_stream_completion,
    stream_response,
    track_request,
)

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
        self.tracer = self._initialize_tracer(enable_tracing)

    def __getattr__(
        self,
        name: Union[Literal["openai"], Literal["anthropic"], Literal["prompts"]],
    ):
        if name == "openai":
            import openai as openai_module

            openai = PromptLayerBase(
                openai_module,
                function_name="openai",
                api_key=self.api_key,
                tracer=self.tracer,
            )
            return openai
        elif name == "anthropic":
            import anthropic as anthropic_module

            anthropic = PromptLayerBase(
                anthropic_module,
                function_name="anthropic",
                provider_type="anthropic",
                api_key=self.api_key,
                tracer=self.tracer,
            )
            return anthropic
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")

    def _initialize_tracer(self, enable_tracing: bool):
        if enable_tracing:
            resource = Resource(
                attributes={ResourceAttributes.SERVICE_NAME: "prompt-layer-library"}
            )
            tracer_provider = TracerProvider(resource=resource)
            promptlayer_exporter = PromptLayerSpanExporter(api_key=self.api_key)
            span_processor = BatchSpanProcessor(promptlayer_exporter)
            tracer_provider.add_span_processor(span_processor)
            trace.set_tracer_provider(tracer_provider)
            return trace.get_tracer(__name__)
        else:
            return None

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

        llm_request_span_id = None

        # Make the request
        if self.tracer:
            with self.tracer.start_as_current_span("llm_request") as llm_request_span:
                llm_request_span_id = hex(llm_request_span.context.span_id)[2:].zfill(
                    16
                )
                llm_request_span.set_attribute("provider", provider)
                llm_request_span.set_attribute("function_name", function_name)
                response = request_function(prompt_blueprint, **kwargs)
        else:
            response = request_function(prompt_blueprint, **kwargs)

        # Define tracking function
        def _track_request(**body):
            request_end_time = datetime.datetime.now(datetime.timezone.utc).timestamp()

            if self.tracer:
                with self.tracer.start_as_current_span("track_request"):
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
                        span_id=llm_request_span_id,
                        **body,
                    )
            else:
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
                    span_id=llm_request_span_id,
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

    def traceable(self, metadata=None):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.tracer:
                    with self.tracer.start_as_current_span(func.__name__) as span:
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
