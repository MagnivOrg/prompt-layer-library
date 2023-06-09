import os
import sys

import promptlayer.langchain as langchain
import promptlayer.prompts as prompts
import promptlayer.track as track
from promptlayer.promptlayer import PromptLayerBase

api_key = os.environ.get("PROMPTLAYER_API_KEY")


def get_openai():
    import openai as openai_module

    openai = PromptLayerBase(openai_module, function_name="openai")
    return openai


def get_anthropic():
    import anthropic as anthropic_module

    anthropic = PromptLayerBase(
        anthropic_module,
        function_name="anthropic",
        provider_type="anthropic",
    )
    return anthropic


__all__ = ["api_key", "get_openai", "get_anthropic"]
