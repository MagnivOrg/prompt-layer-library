import datetime
import os
from copy import deepcopy
from typing import Any, Dict, List, Literal, Union

from promptlayer.groups import GroupManager
from promptlayer.promptlayer import PromptLayerBase
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
    def __init__(self, api_key: str = None):
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
        stream=False,
    ):
        template_get_params: GetPromptTemplate = {}
        if prompt_version:
            template_get_params["version"] = prompt_version
        if prompt_release_label:
            template_get_params["label"] = prompt_release_label
        if input_variables:
            template_get_params["input_variables"] = input_variables
        if metadata:
            template_get_params["metadata_filters"] = metadata
        prompt_blueprint = self.templates.get(prompt_name, template_get_params)
        prompt_template = prompt_blueprint["prompt_template"]
        if not prompt_blueprint["llm_kwargs"]:
            raise Exception(
                f"Prompt '{prompt_name}' does not have any LLM kwargs associated with it."
            )
        prompt_blueprint_metadata = prompt_blueprint.get("metadata", None)
        if prompt_blueprint_metadata is None:
            raise Exception(
                f"Prompt '{prompt_name}' does not have any metadata associated with it."
            )
        prompt_blueprint_model = prompt_blueprint_metadata.get("model", None)
        if prompt_blueprint_model is None:
            raise Exception(
                f"Prompt '{prompt_name}' does not have a model parameters associated with it."
            )
        provider = prompt_blueprint_model["provider"]
        request_start_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
        kwargs = deepcopy(prompt_blueprint["llm_kwargs"])
        config = MAP_PROVIDER_TO_FUNCTION_NAME[provider][prompt_template["type"]]
        function_name = config["function_name"]
        stream_function = config["stream_function"]
        request_function = MAP_PROVIDER_TO_FUNCTION[provider]
        provider_base_url = prompt_blueprint.get("provider_base_url", None)
        if provider_base_url:
            kwargs["base_url"] = provider_base_url["url"]
        kwargs["stream"] = stream
        if stream and provider == "openai":
            kwargs["stream_options"] = {"include_usage": True}
        response = request_function(prompt_blueprint, **kwargs)

        def _track_request(**body):
            request_end_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
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

        if stream:
            return stream_response(response, _track_request, stream_function)
        request_log = _track_request(request_response=response.model_dump())
        data = {
            "request_id": request_log["request_id"],
            "raw_response": response,
            "prompt_blueprint": request_log["prompt_blueprint"],
        }
        return data


__version__ = "1.0.9"
__all__ = ["PromptLayer", "__version__"]
