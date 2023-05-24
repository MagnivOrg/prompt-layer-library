import os

import anthropic
import openai

import promptlayer.langchain as langchain
import promptlayer.prompts as prompts
import promptlayer.track as track
from promptlayer.promptlayer import PromptLayerBase

api_key = os.environ.get("PROMPTLAYER_API_KEY")
openai = PromptLayerBase(openai, function_name="openai")

anthropic_client = PromptLayerBase(
    anthropic.Client(""),
    function_name="anthropic",
    provider_type="anthropic",
)


__all__ = ["api_key", "openai", "anthropic_client"]
