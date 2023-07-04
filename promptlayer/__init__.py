import os
from typing import Literal, Union

import promptlayer.langchain as langchain
import promptlayer.prompts as prompts
import promptlayer.track as track
from promptlayer.promptlayer import PromptLayerBase

api_key = os.environ.get("PROMPTLAYER_API_KEY")


def __getattr__(name: Union[Literal["openai"], Literal["anthropic"]]):
    if name == "openai":
        import openai as openai_module

        openai = PromptLayerBase(openai_module, function_name="openai")
        return openai
    elif name == "anthropic":
        import anthropic as anthropic_module

        anthropic = PromptLayerBase(
            anthropic_module,
            function_name="anthropic",
            provider_type="anthropic",
        )
        return anthropic
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["api_key", "openai", "anthropic"]
