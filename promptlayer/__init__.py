import os
from typing import Literal, Union

from promptlayer.groups import GroupManager
from promptlayer.promptlayer import PromptLayerBase
from promptlayer.templates import TemplateManager
from promptlayer.track import TrackManager


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


__version__ = "1.0.2"
__all__ = ["PromptLayer", "__version__"]
