import asyncio
import datetime
from copy import deepcopy
from functools import wraps
from typing import Any, Dict, Union

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from promptlayer.span_exporter import PromptLayerSpanExporter
from promptlayer.streaming import (
    aanthropic_stream_completion,
    aanthropic_stream_message,
    abedrock_stream_message,
    agoogle_stream_chat,
    agoogle_stream_completion,
    amistral_stream_chat,
    anthropic_stream_completion,
    anthropic_stream_message,
    aopenai_responses_stream_chat,
    aopenai_stream_chat,
    aopenai_stream_completion,
    bedrock_stream_message,
    google_stream_chat,
    google_stream_completion,
    mistral_stream_chat,
    openai_responses_stream_chat,
    openai_stream_chat,
    openai_stream_completion,
)
from promptlayer.utils import (
    aamazon_bedrock_request,
    aanthropic_bedrock_request,
    aanthropic_request,
    aazure_openai_request,
    agoogle_request,
    amazon_bedrock_request,
    amistral_request,
    anthropic_bedrock_request,
    anthropic_request,
    aopenai_request,
    avertexai_request,
    azure_openai_request,
    google_request,
    mistral_request,
    openai_request,
    vertexai_request,
)

MAP_PROVIDER_TO_FUNCTION_NAME = {
    "openai:chat-completions": {
        "chat": {
            "function_name": "openai.chat.completions.create",
            "stream_function": openai_stream_chat,
        },
        "completion": {
            "function_name": "openai.completions.create",
            "stream_function": openai_stream_completion,
        },
    },
    "openai:responses": {
        "chat": {
            "function_name": "openai.responses.create",
            "stream_function": openai_responses_stream_chat,
        },
        "completion": {
            "function_name": "openai.responses.create",
            "stream_function": openai_responses_stream_chat,
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
    "openai.azure:chat-completions": {
        "chat": {
            "function_name": "openai.AzureOpenAI.chat.completions.create",
            "stream_function": openai_stream_chat,
        },
        "completion": {
            "function_name": "openai.AzureOpenAI.completions.create",
            "stream_function": openai_stream_completion,
        },
    },
    "openai.azure:responses": {
        "chat": {
            "function_name": "openai.AzureOpenAI.responses.create",
            "stream_function": openai_responses_stream_chat,
        },
        "completion": {
            "function_name": "openai.AzureOpenAI.responses.create",
            "stream_function": openai_responses_stream_chat,
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
    "google": {
        "chat": {
            "function_name": "google.convo.send_message",
            "stream_function": google_stream_chat,
        },
        "completion": {
            "function_name": "google.model.generate_content",
            "stream_function": google_stream_completion,
        },
    },
    "amazon.bedrock": {
        "chat": {
            "function_name": "boto3.bedrock-runtime.converse",
            "stream_function": bedrock_stream_message,
        },
        "completion": {
            "function_name": "boto3.bedrock-runtime.converse",
            "stream_function": bedrock_stream_message,
        },
    },
    "anthropic.bedrock": {
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
    "anthropic": anthropic_request,
    "google": google_request,
    "mistral": mistral_request,
    "openai": openai_request,
    "openai.azure": azure_openai_request,
    "vertexai": vertexai_request,
    "amazon.bedrock": amazon_bedrock_request,
    "anthropic.bedrock": anthropic_bedrock_request,
}

AMAP_PROVIDER_TO_FUNCTION_NAME = {
    "openai:chat-completions": {
        "chat": {
            "function_name": "openai.chat.completions.create",
            "stream_function": aopenai_stream_chat,
        },
        "completion": {
            "function_name": "openai.completions.create",
            "stream_function": aopenai_stream_completion,
        },
    },
    "openai:responses": {
        "chat": {
            "function_name": "openai.responses.create",
            "stream_function": aopenai_responses_stream_chat,
        },
        "completion": {
            "function_name": "openai.responses.create",
            "stream_function": aopenai_responses_stream_chat,
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
    "openai.azure:chat-completions": {
        "chat": {
            "function_name": "openai.AzureOpenAI.chat.completions.create",
            "stream_function": aopenai_stream_chat,
        },
        "completion": {
            "function_name": "openai.AzureOpenAI.completions.create",
            "stream_function": aopenai_stream_completion,
        },
    },
    "openai.azure:responses": {
        "chat": {
            "function_name": "openai.AzureOpenAI.responses.create",
            "stream_function": aopenai_responses_stream_chat,
        },
        "completion": {
            "function_name": "openai.AzureOpenAI.responses.create",
            "stream_function": aopenai_responses_stream_chat,
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
    "google": {
        "chat": {
            "function_name": "google.convo.send_message",
            "stream_function": agoogle_stream_chat,
        },
        "completion": {
            "function_name": "google.model.generate_content",
            "stream_function": agoogle_stream_completion,
        },
    },
    "amazon.bedrock": {
        "chat": {
            "function_name": "boto3.bedrock-runtime.converse",
            "stream_function": abedrock_stream_message,
        },
        "completion": {
            "function_name": "boto3.bedrock-runtime.converse",
            "stream_function": abedrock_stream_message,
        },
    },
    "anthropic.bedrock": {
        "chat": {
            "function_name": "anthropic.messages.create",
            "stream_function": aanthropic_stream_message,
        },
        "completion": {
            "function_name": "anthropic.completions.create",
            "stream_function": aanthropic_stream_completion,
        },
    },
}


AMAP_PROVIDER_TO_FUNCTION = {
    "anthropic": aanthropic_request,
    "google": agoogle_request,
    "mistral": amistral_request,
    "openai": aopenai_request,
    "openai.azure": aazure_openai_request,
    "vertexai": avertexai_request,
    "amazon.bedrock": aamazon_bedrock_request,
    "anthropic.bedrock": aanthropic_bedrock_request,
}


class PromptLayerMixin:
    @staticmethod
    def _initialize_tracer(api_key: str, base_url: str, enable_tracing: bool = False):
        if enable_tracing:
            resource = Resource(attributes={ResourceAttributes.SERVICE_NAME: "prompt-layer-library"})
            tracer_provider = TracerProvider(resource=resource)
            promptlayer_exporter = PromptLayerSpanExporter(api_key=api_key, base_url=base_url)
            span_processor = BatchSpanProcessor(promptlayer_exporter)
            tracer_provider.add_span_processor(span_processor)
            tracer = tracer_provider.get_tracer(__name__)
            return tracer_provider, tracer
        else:
            return None, None

    @staticmethod
    def _prepare_get_prompt_template_params(
        *,
        prompt_version: Union[int, None],
        prompt_release_label: Union[str, None],
        input_variables: Union[Dict[str, Any], None],
        metadata: Union[Dict[str, str], None],
        provider: Union[str, None] = None,
        model: Union[str, None] = None,
        model_parameter_overrides: Union[Dict[str, Any], None] = None,
    ) -> Dict[str, Any]:
        params = {}

        if prompt_version:
            params["version"] = prompt_version
        if prompt_release_label:
            params["label"] = prompt_release_label
        if input_variables:
            params["input_variables"] = input_variables
        if metadata:
            params["metadata_filters"] = metadata
        if provider:
            params["provider"] = provider
        if model:
            params["model"] = model
        if model_parameter_overrides:
            params["model_parameter_overrides"] = model_parameter_overrides

        return params

    @staticmethod
    def _prepare_llm_data(
        *,
        prompt_blueprint,
        prompt_template,
        prompt_blueprint_model,
        stream,
        is_async=False,
    ):
        client_kwargs = {}
        function_kwargs = deepcopy(prompt_blueprint["llm_kwargs"])
        function_kwargs["stream"] = stream
        provider = prompt_blueprint_model["provider"]
        api_type = prompt_blueprint_model.get("api_type", "chat-completions")

        if custom_provider := prompt_blueprint.get("custom_provider"):
            provider = custom_provider["client"]
            client_kwargs = {
                "api_key": custom_provider["api_key"],
                "base_url": custom_provider["base_url"],
            }
        elif provider_base_url := prompt_blueprint.get("provider_base_url"):
            client_kwargs["base_url"] = provider_base_url["url"]

        if stream and provider in ["openai", "openai.azure"] and api_type == "chat-completions":
            function_kwargs["stream_options"] = {"include_usage": True}

        provider_function_name = provider
        if provider_function_name == "vertexai":
            if "gemini" in prompt_blueprint_model["name"]:
                provider_function_name = "google"
            elif "claude" in prompt_blueprint_model["name"]:
                provider_function_name = "anthropic"

        if provider_function_name in ("openai", "openai.azure"):
            provider_function_name = f"{provider_function_name}:{api_type}"

        if is_async:
            config = AMAP_PROVIDER_TO_FUNCTION_NAME[provider_function_name][prompt_template["type"]]
            request_function = AMAP_PROVIDER_TO_FUNCTION[provider]
        else:
            config = MAP_PROVIDER_TO_FUNCTION_NAME[provider_function_name][prompt_template["type"]]
            request_function = MAP_PROVIDER_TO_FUNCTION[provider]

        return {
            "provider": provider,
            "function_name": config["function_name"],
            "stream_function": config["stream_function"],
            "request_function": request_function,
            "client_kwargs": client_kwargs,
            "function_kwargs": function_kwargs,
            "prompt_blueprint": prompt_blueprint,
        }

    @staticmethod
    def _validate_and_extract_model_from_prompt_blueprint(*, prompt_blueprint, prompt_name):
        if not prompt_blueprint["llm_kwargs"]:
            raise ValueError(
                f"Prompt '{prompt_name}' does not have any LLM kwargs associated with it. Please set your model parameters in the registry in the PromptLayer dashbaord."
            )

        prompt_blueprint_metadata = prompt_blueprint.get("metadata")

        if not prompt_blueprint_metadata:
            raise ValueError(f"Prompt '{prompt_name}' does not have any metadata associated with it.")

        prompt_blueprint_model = prompt_blueprint_metadata.get("model")

        if not prompt_blueprint_model:
            raise ValueError(f"Prompt '{prompt_name}' does not have a model parameters associated with it.")

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
            "kwargs": request_params["function_kwargs"],
            "tags": tags,
            "request_start_time": datetime.datetime.now(datetime.timezone.utc).timestamp(),
            "request_end_time": datetime.datetime.now(datetime.timezone.utc).timestamp(),
            "api_key": api_key,
            "metadata": metadata,
            "prompt_id": request_params["prompt_blueprint"]["id"],
            "prompt_version": request_params["prompt_blueprint"]["version"],
            "prompt_input_variables": input_variables or {},
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

                        span.set_attribute("function_input", str({"args": args, "kwargs": kwargs}))
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

                        span.set_attribute("function_input", str({"args": args, "kwargs": kwargs}))
                        result = await func(*args, **kwargs)
                        span.set_attribute("function_output", str(result))

                        return result
                else:
                    return await func(*args, **kwargs)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator
