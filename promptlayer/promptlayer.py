import asyncio
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
    def __init__(
        self,
        api_key: str = None,
        enable_tracing: bool = False,
        workspace_id: int = None,
    ):
        if api_key is None:
            api_key = os.environ.get("PROMPTLAYER_API_KEY")

        if api_key is None:
            raise ValueError(
                "PromptLayer API key not provided. "
                "Please set the PROMPTLAYER_API_KEY environment variable or pass the api_key parameter."
            )

        if enable_tracing and not workspace_id:
            raise ValueError("Please set a workspace_id to enable tracing.")

        self.api_key = api_key
        self.templates = TemplateManager(api_key)
        self.group = GroupManager(api_key)
        self.tracer = self._initialize_tracer(api_key, enable_tracing, workspace_id)
        self.track = TrackManager(api_key)
        self.workspace_id = workspace_id

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

    def _create_track_request_callable(
        self, request_params, tags, input_variables, group_id, span_id
    ):
        def _track_request(**body):
            track_request_kwargs = self._prepare_track_request_kwargs(
                request_params, tags, input_variables, group_id, span_id, **body
            )
            if self.tracer:
                with self.tracer.start_as_current_span("track_request"):
                    return track_request(**track_request_kwargs)
            return track_request(**track_request_kwargs)

        return _track_request

    def _fetch_prompt_blueprint(self, *, prompt_name, template_params):
        if self.tracer:
            with self.tracer.start_as_current_span("fetch_prompt_template") as span:
                span.set_attribute("prompt_name", prompt_name)
                span.set_attribute(
                    "function_input",
                    str(
                        {"prompt_name": prompt_name, "template_params": template_params}
                    ),
                )
                result = self.templates.get(prompt_name, template_params)
                span.set_attribute("function_output", str(result))
                return result
        return self.templates.get(prompt_name, template_params)

    @staticmethod
    def _initialize_tracer(
        api_key: str = None, enable_tracing: bool = False, workspace_id: int = None
    ):
        if enable_tracing:
            resource = Resource(
                attributes={ResourceAttributes.SERVICE_NAME: "prompt-layer-library"}
            )
            tracer_provider = TracerProvider(resource=resource)
            promptlayer_exporter = PromptLayerSpanExporter(
                api_key=api_key, workspace_id=workspace_id
            )
            span_processor = BatchSpanProcessor(promptlayer_exporter)
            tracer_provider.add_span_processor(span_processor)
            trace.set_tracer_provider(tracer_provider)
            return trace.get_tracer(__name__)
        else:
            return None

    def _make_llm_request(self, request_params):
        span_id = None

        if self.tracer:
            with self.tracer.start_as_current_span("llm_request") as span:
                span.set_attribute("provider", request_params["provider"])
                span.set_attribute("function_name", request_params["function_name"])
                span.set_attribute("function_input", str(request_params))
                span_id = hex(span.context.span_id)[2:].zfill(16)
                response = request_params["request_function"](
                    request_params["prompt_blueprint"], **request_params["kwargs"]
                )
                span.set_attribute("function_output", str(response))
        else:
            response = request_params["request_function"](
                request_params["prompt_blueprint"], **request_params["kwargs"]
            )

        return response, span_id

    @staticmethod
    def _prepare_get_prompt_template_params(
        *, prompt_version, prompt_release_label, input_variables, metadata
    ):
        params = {}

        if prompt_version:
            params["version"] = prompt_version
        if prompt_release_label:
            params["label"] = prompt_release_label
        if input_variables:
            params["input_variables"] = input_variables
        if metadata:
            params["metadata_filters"] = metadata

        return params

    @staticmethod
    def _prepare_llm_request_params(
        *, prompt_blueprint, prompt_template, prompt_blueprint_model, stream
    ):
        provider = prompt_blueprint_model["provider"]
        kwargs = deepcopy(prompt_blueprint["llm_kwargs"])
        config = MAP_PROVIDER_TO_FUNCTION_NAME[provider][prompt_template["type"]]

        if provider_base_url := prompt_blueprint.get("provider_base_url"):
            kwargs["base_url"] = provider_base_url["url"]

        kwargs["stream"] = stream
        if stream and provider == "openai":
            kwargs["stream_options"] = {"include_usage": True}

        return {
            "provider": provider,
            "function_name": config["function_name"],
            "stream_function": config["stream_function"],
            "request_function": MAP_PROVIDER_TO_FUNCTION[provider],
            "kwargs": kwargs,
            "prompt_blueprint": prompt_blueprint,
        }

    def _prepare_track_request_kwargs(
        self, request_params, tags, input_variables, group_id, span_id, **body
    ):
        return {
            "function_name": request_params["function_name"],
            "provider_type": request_params["provider"],
            "args": [],
            "kwargs": request_params["kwargs"],
            "tags": tags,
            "request_start_time": datetime.datetime.now(
                datetime.timezone.utc
            ).timestamp(),
            "request_end_time": datetime.datetime.now(
                datetime.timezone.utc
            ).timestamp(),
            "api_key": self.api_key,
            "metadata": request_params.get("metadata"),
            "prompt_id": request_params["prompt_blueprint"]["id"],
            "prompt_version": request_params["prompt_blueprint"]["version"],
            "prompt_input_variables": input_variables,
            "group_id": group_id,
            "return_prompt_blueprint": True,
            "span_id": span_id,
            **body,
        }

    def _run_internal(
        self,
        *,
        prompt_name: str,
        prompt_version: Union[int, None] = None,
        prompt_release_label: Union[str, None] = None,
        input_variables: Union[Dict[str, Any], None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, str], None] = None,
        group_id: Union[int, None] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        get_prompt_template_params = self._prepare_get_prompt_template_params(
            prompt_version=prompt_version,
            prompt_release_label=prompt_release_label,
            input_variables=input_variables,
            metadata=metadata,
        )
        prompt_blueprint = self._fetch_prompt_blueprint(
            prompt_name=prompt_name, template_params=get_prompt_template_params
        )
        prompt_blueprint_model = self._validate_and_extract_model_from_prompt_blueprint(
            prompt_blueprint=prompt_blueprint, prompt_name=prompt_name
        )
        llm_request_params = self._prepare_llm_request_params(
            prompt_blueprint=prompt_blueprint,
            prompt_template=prompt_blueprint["prompt_template"],
            prompt_blueprint_model=prompt_blueprint_model,
            stream=stream,
        )

        response, span_id = self._make_llm_request(llm_request_params)

        if stream:
            return stream_response(
                response,
                self._create_track_request_callable(
                    llm_request_params, tags, input_variables, group_id, span_id
                ),
                llm_request_params["stream_function"],
            )

        request_log = self._track_request_log(
            llm_request_params,
            tags,
            input_variables,
            group_id,
            span_id,
            request_response=response.model_dump(),
        )

        return {
            "request_id": request_log.get("request_id", None),
            "raw_response": response,
            "prompt_blueprint": request_log.get("prompt_blueprint", None),
        }

    def _track_request_log(
        self, request_params, tags, input_variables, group_id, span_id, **body
    ):
        track_request_kwargs = self._prepare_track_request_kwargs(
            request_params, tags, input_variables, group_id, span_id, **body
        )
        if self.tracer:
            with self.tracer.start_as_current_span("track_request") as span:
                span.set_attribute("function_input", str(track_request_kwargs))
                result = track_request(**track_request_kwargs)
                span.set_attribute("function_output", str(result))
                return result
        return track_request(**track_request_kwargs)

    @staticmethod
    def _validate_and_extract_model_from_prompt_blueprint(
        *, prompt_blueprint, prompt_name
    ):
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

        return prompt_blueprint_model

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
        _run_internal_kwargs = {
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "prompt_release_label": prompt_release_label,
            "input_variables": input_variables,
            "tags": tags,
            "metadata": metadata,
            "group_id": group_id,
            "stream": stream,
        }

        if self.tracer:
            with self.tracer.start_as_current_span("PromptLayer.run") as main_span:
                main_span.set_attribute("prompt_name", prompt_name)
                main_span.set_attribute("stream", stream)
                main_span.set_attribute("function_input", str(_run_internal_kwargs))
                result = self._run_internal(**_run_internal_kwargs)
                main_span.set_attribute("function_output", str(result))
                return result
        else:
            return self._run_internal(**_run_internal_kwargs)

    def traceable(self, metadata=None):
        def decorator(func):
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if self.tracer:
                    with self.tracer.start_as_current_span(func.__name__) as span:
                        if metadata:
                            for key, value in metadata.items():
                                span.set_attribute(key, value)

                        span.set_attribute(
                            "function_input", str({"args": args, "kwargs": kwargs})
                        )
                        result = func(*args, **kwargs)
                        span.set_attribute("function_output", str(result))

                        return result
                else:
                    return func(*args, **kwargs)

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if self.tracer:
                    with self.tracer.start_as_current_span(func.__name__) as span:
                        if metadata:
                            for key, value in metadata.items():
                                span.set_attribute(key, value)

                        span.set_attribute(
                            "function_input", str({"args": args, "kwargs": kwargs})
                        )
                        result = await func(*args, **kwargs)
                        span.set_attribute("function_output", str(result))

                        return result
                else:
                    return await func(*args, **kwargs)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator
