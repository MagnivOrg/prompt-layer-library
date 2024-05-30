import datetime
import os
from copy import deepcopy
from typing import Dict, List, Literal, Union

from promptlayer.groups import GroupManager
from promptlayer.promptlayer import PromptLayerBase
from promptlayer.templates import TemplateManager
from promptlayer.track import TrackManager
from promptlayer.types.prompt_template import GetPromptTemplate
from promptlayer.utils import anthropic_request, openai_request, track_request

MAP_PROVIDER_TO_FUNCTION_NAME = {
    "openai": {
        "chat": "openai.chat.completions.create",
        "completion": "openai.completions.create",
    },
    "anthropic": {
        "chat": "anthropic.messages.create",
        "completion": "anthropic.completions.create",
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
        version: Union[int, None] = None,
        label: Union[str, None] = None,
        input_variables: Union[Dict[str, str], None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, str], None] = None,
        group_id: Union[int, None] = None,
    ):
        template_get_params: GetPromptTemplate = {}
        if version:
            template_get_params["version"] = version
        if label:
            template_get_params["label"] = label
        if input_variables:
            template_get_params["input_variables"] = input_variables
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
        function_name = MAP_PROVIDER_TO_FUNCTION_NAME[provider][prompt_template["type"]]
        response = MAP_PROVIDER_TO_FUNCTION[provider](prompt_blueprint, **kwargs)
        response = response.model_dump()

        request_end_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
        request_log = track_request(
            function_name=function_name,
            provider=provider,
            args=[],
            kwargs=kwargs,
            tags=tags,
            request_response=response,
            request_start_time=request_start_time,
            request_end_time=request_end_time,
            api_key=self.api_key,
            metadata=metadata,
            prompt_id=prompt_blueprint["id"],
            prompt_version=prompt_blueprint["version"],
            prompt_input_variables=input_variables,
            group_id=group_id,
            return_data=True,
        )
        return request_log


__version__ = "1.0.2"
__all__ = ["PromptLayer", "__version__"]
