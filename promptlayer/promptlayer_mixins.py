import asyncio
import datetime
from copy import deepcopy
from functools import wraps
from typing import Dict, Union

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from promptlayer.span_exporter import PromptLayerSpanExporter
from promptlayer.utils import (
    aanthropic_request,
    aanthropic_stream_completion,
    aanthropic_stream_message,
    aazure_openai_request,
    amistral_request,
    amistral_stream_chat,
    anthropic_request,
    anthropic_stream_completion,
    anthropic_stream_message,
    aopenai_request,
    aopenai_stream_chat,
    aopenai_stream_completion,
    azure_openai_request,
    mistral_request,
    mistral_stream_chat,
    openai_request,
    openai_stream_chat,
    openai_stream_completion,
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
    "openai.azure": {
        "chat": {
            "function_name": "openai.AzureOpenAI.chat.completions.create",
            "stream_function": openai_stream_chat,
        },
        "completion": {
            "function_name": "openai.AzureOpenAI.completions.create",
            "stream_function": openai_stream_completion,
        },
    },
    "mistral": {
        "chat": {
            "function_name": "mistral.client.chat",
            "stream_function": mistral_stream_chat,
        },
        "completion": {
            "function_name": None,
            "stream_function": None,
        },
    },
}


MAP_PROVIDER_TO_FUNCTION = {
    "openai": openai_request,
    "anthropic": anthropic_request,
    "openai.azure": azure_openai_request,
    "mistral": mistral_request,
}

AMAP_PROVIDER_TO_FUNCTION_NAME = {
    "openai": {
        "chat": {
            "function_name": "openai.chat.completions.create",
            "stream_function": aopenai_stream_chat,
        },
        "completion": {
            "function_name": "openai.completions.create",
            "stream_function": aopenai_stream_completion,
        },
    },
    "anthropic": {
        "chat": {
            "function_name": "anthropic.messages.create",
            "stream_function": aanthropic_stream_message,
        },
        "completion": {
            "function_name": "anthropic.completions.create",
            "stream_function": aanthropic_stream_completion,
        },
    },
    "openai.azure": {
        "chat": {
            "function_name": "openai.AzureOpenAI.chat.completions.create",
            "stream_function": aopenai_stream_chat,
        },
        "completion": {
            "function_name": "openai.AzureOpenAI.completions.create",
            "stream_function": aopenai_stream_completion,
        },
    },
    "mistral": {
        "chat": {
            "function_name": "mistral.client.chat",
            "stream_function": amistral_stream_chat,
        },
        "completion": {
            "function_name": None,
            "stream_function": None,
        },
    },
}


AMAP_PROVIDER_TO_FUNCTION = {
    "openai": aopenai_request,
    "anthropic": aanthropic_request,
    "openai.azure": aazure_openai_request,
    "mistral": amistral_request,
}


class PromptLayerMixin:
    @staticmethod
    def _initialize_tracer(api_key: str = None, enable_tracing: bool = False):
        if enable_tracing:
            resource = Resource(
                attributes={ResourceAttributes.SERVICE_NAME: "prompt-layer-library"}
            )
            tracer_provider = TracerProvider(resource=resource)
            promptlayer_exporter = PromptLayerSpanExporter(api_key=api_key)
            span_processor = BatchSpanProcessor(promptlayer_exporter)
            tracer_provider.add_span_processor(span_processor)
            tracer = tracer_provider.get_tracer(__name__)
            return tracer_provider, tracer
        else:
            return None, None

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
        *,
        prompt_blueprint,
        prompt_template,
        prompt_blueprint_model,
        model_parameter_overrides,
        stream,
        is_async=False,
    ):
        provider = prompt_blueprint_model["provider"]
        kwargs = deepcopy(prompt_blueprint["llm_kwargs"])
        if is_async:
            config = AMAP_PROVIDER_TO_FUNCTION_NAME[provider][prompt_template["type"]]
            request_function = AMAP_PROVIDER_TO_FUNCTION[provider]
        else:
            config = MAP_PROVIDER_TO_FUNCTION_NAME[provider][prompt_template["type"]]
            request_function = MAP_PROVIDER_TO_FUNCTION[provider]

        if provider_base_url := prompt_blueprint.get("provider_base_url"):
            kwargs["base_url"] = provider_base_url["url"]

        if model_parameter_overrides:
            kwargs.update(model_parameter_overrides)

        kwargs["stream"] = stream
        if stream and provider in ["openai", "openai.azure"]:
            kwargs["stream_options"] = {"include_usage": True}

        return {
            "provider": provider,
            "function_name": config["function_name"],
            "stream_function": config["stream_function"],
            "request_function": request_function,
            "kwargs": kwargs,
            "prompt_blueprint": prompt_blueprint,
        }

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

    @staticmethod
    def _prepare_track_request_kwargs(
        api_key,
        request_params,
        tags,
        input_variables,
        group_id,
        pl_run_span_id: Union[str, None] = None,
        metadata: Union[Dict[str, str], None] = None,
        **body,
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
            "api_key": api_key,
            "metadata": metadata,
            "prompt_id": request_params["prompt_blueprint"]["id"],
            "prompt_version": request_params["prompt_blueprint"]["version"],
            "prompt_input_variables": input_variables,
            "group_id": group_id,
            "return_prompt_blueprint": True,
            "span_id": pl_run_span_id,
            **body,
        }

    def traceable(self, attributes=None, name=None):
        def decorator(func):
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if self.tracer:
                    span_name = name or func.__name__
                    with self.tracer.start_as_current_span(span_name) as span:
                        if attributes:
                            for key, value in attributes.items():
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
                    span_name = name or func.__name__
                    with self.tracer.start_as_current_span(span_name) as span:
                        if attributes:
                            for key, value in attributes.items():
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
